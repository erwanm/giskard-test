from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn
import argparse

# GPT2
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
# MT5
#from transformers import MT5ForConditionalGeneration, T5Tokenizer
#model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
#tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")

from transformers import pipeline

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token=tokenizer.eos_token

# Set the device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    top_k=None
)



def generate_some_sentences(model, tokenizer, prompt, n, max_length=35, temperature=0.95, topk=200):

    print(f"Generating {n} sentences... ")
    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate text
    generated_sentences = []
    for sample in range(n):
        print(f"\r {sample}/{n} ",end='')
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=topk, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
        generated_sentences.append(generated_text)
    print()
    return generated_sentences



def format_sentiment_single(sentiment):
    return {el["label"]: el["score"] for el in sentiment}



def apply_sentiment_model(sentences):
    sentiments = distilled_student_sentiment_classifier(sentences)
    return [ format_sentiment_single(datum) for datum in sentiments ]



def custom_loss(logits, labels, target_sentiment):

    # Apply softmax to obtain probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    # Sample from the distribution to get predicted tokens
    predicted_token_ids = torch.multinomial(probabilities[:, -1, :], 1)
    print(logits.shape, probabilities.shape, predicted_token_ids.shape)
    # Decode the predicted token IDs to obtain the actual tokens                       
    tokens = [tokenizer.decode(sent_token_ids.squeeze().item(), skip_special_tokens=True) for sent_token_ids in predicted_token_ids]
    for tokens0 in tokens:
        print(" ".join(tokens0)) 
    # Compute the sentiment of the generated sentences
    sentiment_outputs = apply_sentiment_model(tokens)
    a=[]
    for senti in sentiment_outputs:
        for k,score in senti.items():
            a.append(abs(target_sentiment.get(k) - score)) 
    loss = torch.tensor(a, requires_grad=True).sum()
    return loss


def custom_loss2(generated_sentences, target_sentiment):

    # Compute the sentiment of the generated sentences
    sentiment_outputs = apply_sentiment_model(generated_sentences)
    a=[]
    for senti in sentiment_outputs:
        for k,score in senti.items():
            a.append(abs(target_sentiment.get(k) - score)) 
    loss = torch.tensor(a, requires_grad=True).sum()
    return loss




def fine_tune(training_data, sentiment_target, num_epochs, lr, batch_size, num_tokens_to_generate=15):

    encoded_texts_in = tokenizer(training_data, return_tensors='pt', truncation=True, padding=True)

    dataset = TensorDataset(encoded_texts_in['input_ids'], encoded_texts_in['attention_mask'], encoded_texts_in['input_ids'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Fine-tuning loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()

            input_ids, attention_mask, labels = batch

            new_tokens = torch.empty((batch_size,1))
            for _ in range(num_tokens_to_generate):
                outputs = model(input_ids)
                logits = outputs.logits
                next_token_logits = logits[:, -1, :]
                # Generate the next token using top-k sampling
                next_token = torch.multinomial(torch.nn.functional.softmax(next_token_logits / 0.8, dim=-1), 1)
                # Append the new token to the input sequences
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                new_tokens = torch.cat([new_tokens, next_token], dim=-1)

            #print(new_tokens.shape)
            generated_sentences=[]
            for seq in new_tokens:
                seq[seq < 0] = tokenizer.unk_token_id
                seq[seq >= tokenizer.vocab_size] = tokenizer.unk_token_id
                print(seq)
                generated_sentences.append(tokenizer.decode(seq.tolist(), skip_special_tokens=True))
#            print(" ".join([str(l) for l in [ len(seq) for seq in new_tokens]]))
#            generated_sentences = [tokenizer.decode(seq.tolist()) for seq in new_tokens]
            #for tokens in generated_sentences:
            #    print(tokens) 

#            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#            loss = custom_loss(outputs.logits, labels,  sentiment_target)
            loss = custom_loss2(generated_sentences, sentiment_target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_model")




def mymain():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_string", help="input sentence")
    parser.add_argument("--sample_size", help="number of sentences to generate", type=int, default=100)
    parser.add_argument("--temperature", help="temperature for generation (higher=more diverse)", type=int, default=100)
    parser.add_argument("--top_k", help="top_k for generation  (higher=more diverse) ", type=int, default=100)
    parser.add_argument("--num_epochs", help="number of epochs (default 10)", type=int, default=10)
    parser.add_argument("--learn_rate", help="learning rate (default 5e-5) ", type=float, default=5e-5)
    parser.add_argument("--batch_size", help="batch size (default 10) ", type=int, default=10)
    args = parser.parse_args()

    target_sentiment0 = apply_sentiment_model([args.input_string])
    target_sentiment = target_sentiment0[0]

    data = generate_some_sentences(model, tokenizer, args.input_string, args.sample_size)
    for i,t in enumerate(data):
        print(f"{i}: {t}")

    fine_tune(data, target_sentiment, args.num_epochs, args.learn_rate, args.batch_size)



if __name__ == '__main__':
    mymain()

