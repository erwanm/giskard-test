import numpy as np
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



def generate_some_sentences(model, tokenizer, prompt, n, max_length=12, temperature=0.95, topk=200, verbose=False):

    if verbose:
        print(f"Generating {n} sentences, max_length={max_length}... ")
    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = len(input_ids[0])
    
    # Generate text
    generated_sentences = []
    for sample in range(n):
        if verbose:
            print(f"\r {sample}/{n} ",end='')
        output = model.generate(input_ids, max_length=input_length+max_length, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=topk, pad_token_id=tokenizer.eos_token_id)
        #print(output)
        #generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        #print(f"  GENERATED1 = {generated_text}")
        generated_text = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
        #print(f"  GENERATED2 = {generated_text}")
        generated_sentences.append(generated_text)
    if verbose:
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
    #for i,s in enumerate(generated_sentences):
    #    print(f"LOSS {i} {s} {sentiment_outputs[i]}")
    loss = []
    for senti in sentiment_outputs:
        a=[]
        for k,score in senti.items():
            a.append(abs(target_sentiment.get(k) - score)) 
        loss.append(sum(a)/len(a))
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
    parser.add_argument("--num_epochs", help="number of epochs (default 100)", type=int, default=100)
    parser.add_argument("--learn_rate", help="learning rate (default 5e-5) ", type=float, default=5e-5)
    parser.add_argument("--batch_size", help="batch size (default 10) ", type=int, default=10)
    parser.add_argument("--prop_seeds", help="prop seeds ", type=int, default=.05)
    args = parser.parse_args()

    target_sentiment0 = apply_sentiment_model([args.input_string])
    target_sentiment = target_sentiment0[0]
    print("Target sentiment: ", target_sentiment)

    sent_seeds = [args.input_string]
    rng = np.random.default_rng()

    for epoch in range(args.num_epochs):
        probs = [1 / len(sent_seeds)] * len(sent_seeds)
        freq_seeds = rng.multinomial(args.sample_size, probs)
        generation = []
        for id_seed, n_seed in enumerate(freq_seeds):
            data = generate_some_sentences(model, tokenizer, sent_seeds[id_seed], n_seed)
            generation.extend(data)
        if len(generation) != args.sample_size:
            raise Exception(f"Bug: expected {args.sample_size} sentences, found {len(generation)}")
        loss = custom_loss(generation, target_sentiment)
        gen_loss = sum(loss)/len(loss)
        sorted_loss  = np.argsort(loss)
        sent_seeds_indexes = sorted_loss[0:int(args.prop_seeds* args.sample_size)]
        sent_seeds = [ generation[idx] for idx in sent_seeds_indexes ]
        #for pos,idx in enumerate(sent_seeds_indexes):
        #    print(f"SEED  {pos}. {generation[idx]} - loss={loss[idx]}")
        print(f"Generation {epoch}: mean loss = {gen_loss}; min loss = {loss[sent_seeds_indexes[0]]} ")
        




if __name__ == '__main__':
    mymain()

