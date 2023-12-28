from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn
import argparse

# GPT2
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
# MT5
#from transformers import MT5ForConditionalGeneration, T5Tokenizer

from transformers import pipeline


class GPT2CustomLoss(nn.Module):
    def __init__(self, config, target_sentiment):
        super(GPT2CustomLoss, self).__init__()
        self.gpt2 = GPT2LMHeadModel(config)
        self.target_sentiment = target_sentiment

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask, labels=labels)

        # Extract logits from the model's output
        logits = outputs.logits
        # Apply softmax to obtain probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        # Sample from the distribution to get predicted tokens
        predicted_token_ids = torch.multinomial(probabilities[:, -1, :], 1)
        # Decode the predicted token IDs to obtain the actual tokens
        tokens = tokenizer.decode(predicted_token_ids.squeeze().item())

        sentiment_outputs = apply_sentiment_model(tokens)
        loss_values = []
        for senti in sentiment_outputs:
            for k,score in senti.items():
                delta = abs(self.target_sentiment.get(k) - score)
                loss_values.append(delta)

        custom_loss = self.custom_loss(logits, labels)

        # Combine GPT-2 loss and custom loss
        total_loss = outputs.loss + self.custom_loss_alpha * custom_loss

        return total_loss


distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    top_k=None
)

# Set the device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"


def init_custom_gpt2_model(target_sentiment):

    #model = GPT2LMHeadModel.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained('gpt2')
    model = GPT2CustomLoss(config, target_sentiment)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token=tokenizer.eos_token

    #model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    #tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")

    model.to(device)

    return (model, tokenizer)



def generate_some_sentences(model, tokenizer, prompt, n, max_length=35, temperature=0.95, topk=200):

    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate text
    generated_sentences = []
    for sample in range(n):
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=topk, pad_token_id=tokenizer.eos_token_id)
        # Decode and print the generated text
        generated_text = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
        generated_sentences.append(generated_text)
    return generated_sentences



# Custom loss function (example, replace with your own)
def custom_lossBAK(outputs, target_sentiment):

    print(target_sentiment)
    sentiment_outputs = apply_sentiment_model(outputs)
    loss_values = []
    for senti in sentiment_outputs:
        for k,score in senti.items():
            delta = abs(target_sentiment.get(k) - score)
            loss_values.append(delta)
    return sum(loss_values) / len(loss_values)



def fine_tune(input_sentence, sentiment_model, num_epochs, lr, batch_size):

    target_sentiment0 = apply_sentiment_model([input_sentence])
    target_sentiment = target_sentiment0[0]
    model, tokenizer = init_custom_gpt2_model(target_sentiment)

#    texts_in = generate_some_sentences(input_sentence, 12)
#    texts_out = generate_some_sentences(input_sentence, 12)
#    for i,t in enumerate(texts_in):
#        print(f"{i}: {t}") 

#    encoded_texts_in = tokenizer(texts_in, return_tensors='pt', truncation=True, padding=True)
#    encoded_texts_out = tokenizer(texts_out, return_tensors='pt', truncation=True, padding=True)
    # print(encoded_texts)

    # Create a TensorDataset
#    dataset = TensorDataset(encoded_texts_in['input_ids'], encoded_texts_in['attention_mask'], encoded_texts_out['input_ids'])

#    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)


    # Fine-tuning loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        batch = generate_some_sentences(model, tokenizer, input_sentence, batch_size)
        for current_sent in batch:
            optimizer.zero_grad()

            generated = generate_some_sentences(current_sent, 1)

            #input_ids, attention_mask, labels = batch

            #outputs = model(input_ids, labels=labels)
            print(generated)

            #loss = custom_loss(generated)
            loss = model(generated)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / batch_size
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")

    # Save the fine-tuned model
    model.save_pretrained("my_custom_model")



def format_sentiment_single(sentiment):
    return {el["label"]: el["score"] for el in sentiment}



def apply_sentiment_model(sentences):
    sentiments = distilled_student_sentiment_classifier(sentences)
    return [ format_sentiment_single(datum) for datum in sentiments ]



def mymain():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_string", help="input sentence")
    parser.add_argument("--sample_size", help="number of sentences to generate", type=int, default=100)
    parser.add_argument("--temperature", help="temperature for generation (higher=more diverse)", type=int, default=100)
    parser.add_argument("--top_k", help="top_k for generation  (higher=more diverse) ", type=int, default=100)
    parser.add_argument("--num_epochs", help="number of epochs (default 10)", type=int, default=10)
    parser.add_argument("--learn_rate", help="learning rate (default 5e-5) ", type=int, default=5e-5)
    parser.add_argument("--batch_size", help="batch size (default 10) ", type=int, default=10)
    args = parser.parse_args()
    fine_tune(args.input_string, None, args.num_epochs, args.learn_rate, args.batch_size)
#    data = generate_some_sentences(args.input_string, 10)
#    sentiment_data = apply_sentiment_model(data)
#    print(sentiment_data)



if __name__ == '__main__':
    mymain()

