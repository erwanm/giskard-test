from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import random
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



def custom_loss(generated_sentences, target_sentiment):

    # Compute the sentiment of the generated sentences
    sentiment_outputs = apply_sentiment_model(generated_sentences)
    a=[]
    for senti in sentiment_outputs:
        for k,score in senti.items():
            a.append(abs(target_sentiment.get(k) - score)) 
    print(a)
    loss = torch.tensor(a, requires_grad=True).sum()
    return loss




def fine_tune(training_data, sentiment_target, num_epochs, lr, batch_size, num_tokens_to_generate=15):

    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    #optimizer = optim.SGD(model.parameters(), lr=lr)

    # Fine-tuning loop
    for epoch in range(num_epochs):
        model.train()

        optimizer.zero_grad()
        generated_sentences = generate_some_sentences(model, tokenizer, random.choice(training_data), batch_size)
        loss = custom_loss(generated_sentences, sentiment_target)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

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
    print("Target sentiment: ", target_sentiment)

    data = generate_some_sentences(model, tokenizer, args.input_string, args.sample_size)
    for i,t in enumerate(data):
        print(f"{i+1}: {t}")

    fine_tune(data, target_sentiment, args.num_epochs, args.learn_rate, args.batch_size)



if __name__ == '__main__':
    mymain()

