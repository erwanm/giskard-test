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




def mymain():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_string", help="input sentence")
    parser.add_argument("--sample_size", help="number of sentences to generate", type=int, default=100)
    parser.add_argument("--temperature", help="temperature for generation (higher=more diverse)", type=int, default=100)
    parser.add_argument("--top_k", help="top_k for generation  (higher=more diverse) ", type=int, default=100)
    parser.add_argument("--num_epochs", help="number of epochs (default 100)", type=int, default=100)
    parser.add_argument("--prop_seeds", help="prop seeds ", type=int, default=.05)
    args = parser.parse_args()

    epsilon = .0001

    target_sentiment0 = apply_sentiment_model([args.input_string])
    target_sentiment = target_sentiment0[0]
    print("Target sentiment: ", target_sentiment)

    sent_seeds = [args.input_string]
    rng = np.random.default_rng()

    result = None
    for epoch in range(args.num_epochs):
        probs = [1 / len(sent_seeds)] * len(sent_seeds)
        freq_seeds = rng.multinomial(args.sample_size, probs)
        generation = []
        for id_seed, n_seed in enumerate(freq_seeds):
            data = generate_some_sentences(model, tokenizer, sent_seeds[id_seed], n_seed)
            generation.extend(data)
        if len(generation) != args.sample_size:
            raise Exception(f"Bug: expected {args.sample_size} sentences, found {len(generation)}")

        sentiment_outputs = apply_sentiment_model(generation)
        #for i,s in enumerate(generated_sentences):
        #    print(f"LOSS {i} {s} {sentiment_outputs[i]}")
        loss = []
        for idx,senti in enumerate(sentiment_outputs):
            a=[]
            close_enough = True
            for k,score in senti.items():
                a.append(abs(target_sentiment.get(k) - score)) 
                if abs(target_sentiment.get(k) - score) > epsilon:
                    close_enough = False
            loss.append(sum(a)/len(a))
            if close_enough:
                result = (generation[idx],senti)

        gen_loss = sum(loss)/len(loss)
        sorted_loss  = np.argsort(loss)
        sent_seeds_indexes = sorted_loss[0:int(args.prop_seeds* args.sample_size)]
        sent_seeds = [ generation[idx] for idx in sent_seeds_indexes ]
        #for pos,idx in enumerate(sent_seeds_indexes):
        #    print(f"SEED  {pos}. {generation[idx]} - loss={loss[idx]}")
        print(f"{epoch}\tmean\t{gen_loss}")
        print(f"{epoch}\tmin\t{loss[sent_seeds_indexes[0]]}")
        if result is not None:
            break
    
    if result is not None:
        print('Result:')
        print(result[0])
        print(result[1])
    else:
        print('Not found')



if __name__ == '__main__':
    mymain()

