import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()

from datasets import load_dataset

from transformers import AutoTokenizer, pipeline

from trl import AutoModelForCausalLMWithValueHead
from trl import PPOTrainer, PPOConfig
from trl import core

from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim
import torch.nn as nn
import argparse

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


config = {
   "model_name": "lvwerra/gpt2-imdb",
 #   "cls_model_name": "lvwerra/distilbert-imdb",
     "steps": 20000,
    "batch_size": 128,
    "mini_batch_size": 16,
    "ppo_epochs": 200,   
    "txt_in_min_len": 2,
    "txt_in_max_len": 8,
    "txt_out_min_len": 4,
    "txt_out_max_len": 16,
    "lr": 1e-4,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}

ppoconfig = {
    "model_name": "attempt",
    "steps": 20000,
    "batch_size": 128,
    "mini_batch_size": 16,
    "ppo_epochs": 200,   
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}


gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(config['model_name'])
gpt2_model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(config['model_name'])
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
gpt2_tokenizer.pad_token=gpt2_tokenizer.eos_token


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1


gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id
}


sent_kwargs = {
    "function_to_apply": "none",
    "batch_size": config["mini_batch_size"]
}

sentiment_pipe = pipeline(
    top_k= None,
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
)


def format_sentiment_single(sentiment):
    return {el["label"]: el["score"] for el in sentiment}



def apply_sentiment_model(sentences):
    sentiments = sentiment_pipe(sentences, **sent_kwargs)
    return [ format_sentiment_single(datum) for datum in sentiments ]


#text = 'this movie was really bad!!'
#print('Testing sentiment 1: ', text)
#print(sentiment_pipe(text, **sent_kwargs))
#text = 'this movie was really good!!'
#print('Testing sentiment 2: ', text)
#print(sentiment_pipe(text, **sent_kwargs))


gpt2_model.to(device)
gpt2_model_ref.to(device)


 
def generate_some_sentences(model, tokenizer, prompt, n, max_length=35, temperature=0.95, topk=200):

    print(f"Generating {n} sentences... ")
    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = len(input_ids[0])

    # Generate text
    generated_sentences = []
    for sample in range(n):
        print(f"\r {sample}/{n} ",end='')
        nonempty_sent = False
        while not nonempty_sent:
            output = model.generate(input_ids, max_length=input_length+max_length, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=topk, pad_token_id=tokenizer.eos_token_id)
            new_part =output[0, input_ids.shape[-1]:]
            if len(new_part) > 0 and new_part[0] != tokenizer.eos_token_id:
                generated_text = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
                nonempty_sent = True
        generated_sentences.append(generated_text)
    print()
    #for i,s in enumerate(generated_sentences):
    #    print(f"SENT {i}: {s}")
    return generated_sentences



# returns a list of rewards
def custom_rewards(generated_sentences, target_sentiment):
    sentiment_outputs = apply_sentiment_model(generated_sentences)
    rewards = []
    for i,senti in enumerate(sentiment_outputs):
        s = 0.0
        for k,score in senti.items():
            s -= abs(target_sentiment.get(k) - score)
        rewards.append(s/len(senti))
        #print(f"  {i}; {generated_sentences[i]}; {senti}; {rewards[-1]}")
    return rewards



def tokenize(sent):
    res={}
    res["tokens"] = gpt2_tokenizer.encode(sent, return_tensors='pt', truncation=True, padding=True).tolist()[0]
    res["query"] = sent
    return res

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, sample):
        super(MyDataset, self).__init__()
        self.aslist = sample

    def __len__(self):
        return len(self.aslist)

    def __getitem__(self, idx):
        return self.aslist[idx]




input_string = "It's pretty sad that everyone enjoyed the party, even though the light was dim."

target_sentiment0 = apply_sentiment_model([input_string])
target_sentiment = target_sentiment0[0]
print("Target sentiment: ", target_sentiment)

sample = generate_some_sentences(gpt2_model, gpt2_tokenizer, input_string, 1024)
mydataset = MyDataset([tokenize(s) for s in sample])
dataloader = torch.utils.data.DataLoader(mydataset, batch_size=config['batch_size'], collate_fn=collator)

ppo_config = PPOConfig(**ppoconfig)
ppo_trainer = PPOTrainer(ppo_config, gpt2_model, gpt2_model_ref, gpt2_tokenizer, dataset=mydataset)

total_ppo_epochs = int(np.ceil(config["steps"]/config['batch_size']))


for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader))):
    logs, timing = dict(), dict()
    t0 = time.time()
    query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
    
    #### Get response from gpt2
    t = time.time()
    response_tensors = []
    for i in range(len(query_tensors)):
        gen_len = 5
        response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0),
                                       max_new_tokens=gen_len, **gen_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing['time/get_response'] = time.time()-t

    #### Compute sentiment score
    t = time.time()
    rewards = torch.tensor(custom_rewards(batch['response'], target_sentiment)).to(device)
    timing['time/get_sentiment_preds'] = time.time()-t
    
    #### Run PPO step 
    t = time.time()
    #print(type(query_tensors), type(response_tensors), type(rewards) , type(rewards.tolist()) )
    #print(rewards)
    #print(rewards.tolist())
    testing = [ torch.tensor(v, requires_grad=True) for v in rewards.clone().detach().requires_grad_(True) ]
    stats = ppo_trainer.step(query_tensors, response_tensors, testing)
    timing['time/optimization'] = time.time()-t
     
    #### Log everything
    print("MEAN", torch.mean(rewards).cpu().numpy())
    timing['time/epoch'] = time.time()-t0
    table_rows = [list(r) for r in zip(batch['query'], batch['response'], rewards.cpu().tolist())]


# TODO
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



#if __name__ == '__main__':
#    mymain()
