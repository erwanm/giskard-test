import torch
import os
import numpy as np

from datasets import load_dataset

from transformers import AutoTokenizer, pipeline

from trl import AutoModelForCausalLMWithValueHead
from trl import PPOTrainer, PPOConfig
from trl import core

model_name = "lvwerra/gpt2-imdb"

imdb_filter_config = {
    "txt_in_min_len": 2,
    "txt_in_max_len": 8,
    "txt_out_min_len": 4,
    "txt_out_max_len": 16,
}

ppo_config = {
    "model_name": "lvwerra/gpt2-imdb",
    "steps": 20000,
    "batch_size": 256,
    "mini_batch_size": 16,
    "ppo_epochs": 4,   
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}

sent_kwargs = {
    "function_to_apply": "none",
    "batch_size": ppo_config["mini_batch_size"]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1

gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
gpt2_model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

gpt2_tokenizer = AutoTokenizer.from_pretrained(model_name)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

gpt2_model.to(device)
gpt2_model_ref.to(device)

gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id
}



class LengthSampler:
    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))
    def __call__(self):
        return np.random.choice(self.values)
    
input_size = LengthSampler(imdb_filter_config["txt_in_min_len"], imdb_filter_config["txt_in_max_len"])
output_size = LengthSampler(imdb_filter_config["txt_out_min_len"], imdb_filter_config["txt_out_max_len"])

def tokenize(sample):
    sample["tokens"] = gpt2_tokenizer.encode(sample["review"])[:input_size()]
    sample["query"] = gpt2_tokenizer.decode(sample["tokens"])
    return sample


def format_sentiment_single(sentiment):
    return {el["label"]: el["score"] for el in sentiment}


def apply_sentiment_model(sentiment_pipe, sentences):
    sentiments = sentiment_pipe(sentences, **sent_kwargs)
    return [ format_sentiment_single(datum) for datum in sentiments ]


# returns a pair of lists (rewards, max_diff): for each sentence, the reward and the highest abs diff.
def custom_rewards(sentiment_pipe, generated_sentences, target_sentiment):
    sentiment_outputs = apply_sentiment_model(sentiment_pipe, generated_sentences)
    rewards = []
    max_diff = []
    for i,senti in enumerate(sentiment_outputs):
        sum_scores = 0.0
        max_diff_score = None
        for k,score in senti.items():
            diff_score = abs(target_sentiment.get(k) - score)
            sum_scores -= diff_score
            if max_diff_score is None or max_diff_score<diff_score:
                max_diff_score = diff_score
        rewards.append(sum_scores/len(senti))
        max_diff.append(max_diff_score)
        #print(f"  {i}; {generated_sentences[i]}; {senti}; {rewards[-1]}")
    return (rewards, max_diff)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def init_dataset():
    ds = load_dataset('imdb', split='train')
    ds = ds.rename_columns({'text': 'review', 'label': 'sentiment'})
    ds = ds.filter(lambda x: len(x["review"])>200, batched=False)
    #print(ds)
    ds = ds.map(tokenize, batched=False)
    return ds

def find_close_sentiment(sentiment_mode_str="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                         input_string = 'This was an absolutely incredibly horrible story.',
                         epsilon=.01,
                         max_epochs=1000):

    # Init sentiment analysis pipeline
    sentiment_pipe = pipeline("sentiment-analysis",sentiment_mode_str, device=pipe_device, top_k=None)

    # Apply to input sentences
    target_sentiment0 = apply_sentiment_model(sentiment_pipe, [input_string])
    target_sentiment = target_sentiment0[0]
    print(target_sentiment)

    ds = init_dataset()
    dataloader = torch.utils.data.DataLoader(ds, batch_size=ppo_config['batch_size'], collate_fn=collator)

    my_ppo_config = PPOConfig(**ppo_config)
    ppo_trainer = PPOTrainer(my_ppo_config, gpt2_model, gpt2_model_ref, gpt2_tokenizer, dataset=ds)

    for epoch in range(max_epochs):
        for batch_id,batch in enumerate(dataloader):
            query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
    
            #### Get response from gpt2
            response_tensors = []
            for i in range(len(query_tensors)):
                gen_len = output_size()
                response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0), max_new_tokens=gen_len, **gen_kwargs)
                response_tensors.append(response.squeeze()[-gen_len:])
                batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]

            #### Compute sentiment score
            rewards_list, max_diff_list = custom_rewards(sentiment_pipe, batch['response'], target_sentiment)
            # the highest abs diff is used to check that all the scores are close to the target sentiment, since
            # if the highest abs diff is
            idx_closest_sentiment = np.argmin(max_diff_list)
            if max_diff_list[idx_closest_sentiment] <= epsilon:
                result_sentence = batch['response'][idx_closest_sentiment]
                result_sentiment = apply_sentiment_model(sentiment_pipe, [result_sentence])
                return (result_sentence, result_sentiment)
            rewards = torch.tensor(rewards_list).to(device)
    
            #### Run PPO step 
            testing = [ torch.tensor(v, requires_grad=True) for v in rewards.clone().detach().requires_grad_(True) ]
            stats = ppo_trainer.step(query_tensors, response_tensors, testing)
     
            #### Log everything
            print(f"***** EPOCH {epoch}, batch {batch_id}: MEAN = {torch.mean(rewards).cpu().numpy()}, CLOSEST HIGHEST DIFF = {max_diff_list[idx_closest_sentiment]}")



if __name__ == '__main__':
    sentence, sentiment = find_close_sentiment()
    if sentence is None:
        print("Failure :(")
    else:
        print("Success :)")
        print(sentence)
        print(sentiment)


