import torch
import os
import numpy as np

from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead
from trl import PPOTrainer, PPOConfig
from trl import core

from Levenshtein import distance

from .base import BaseProbe, ProbeResult
from chameleon.models import HuggingFaceModel



##### CONFIG

default_epsilon = 1e-3
default_sample_size = 4096
model_name = "lvwerra/gpt2-imdb"

default_ppo_config = {
    "steps": 20000,
    "batch_size": 256,
    "mini_batch_size": 16,
    "ppo_epochs": 4,   
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":1000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
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



##### UTILITY 

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, sample):
        super(MyDataset, self).__init__()
        self.aslist = sample
        
    def __len__(self):
        return len(self.aslist)

    def __getitem__(self, idx):
        return self.aslist[idx]


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])



##### MAIN CLASS

class Gpt2TrlProbe(BaseProbe):
    """A wrapper for the GPT2 TRL probe.
       This probe uses the TRL module (https://github.com/huggingface/trl)
       to tune a GPT2 model into generating sentences for which the
       sentiment is as close as possible to the target sentence.
    """

    def __init__(self, model: str, target: str):
        """Initialize the probe.

        Parameters
        ----------
        model : BaseModel
            A chameleon model instance, e.g. HuggingFaceModel("distilbert-base-uncased-finetuned-sst-2-english")
        """
        self.model = model
        self.target = target


    def run(self, **kwargs):
        self.epsilon = default_epsilon
        if kwargs.get('epsilon') is not None:
            self.epsilon = kwargs.get('epsilon')
        self.sample_size = default_sample_size
        if kwargs.get('sample_size') is not None:
            self.sample_size = kwargs.get('sample_size')
        self.ppo_config = default_ppo_config
        for name, value in self.ppo_config.items():
            if kwargs.get(name) is not None:
                self.ppo_config[name] = kwargs.get(name)
        return self.find_close_sentiment()


    # apply the sentiment model to a list of sentences
    def apply_sentiment_model(self, sentences):
        return [ self.model.predict(sentence) for sentence in sentences ]


    # returns a pair of lists (rewards, max_diff): for each sentence, the reward and the highest abs diff.
    # 
    # The loss for each sentence is the mean across sentiment scores of the absolute difference between
    # this score and the target score (eg. if target 'positive' is 0.7 and for this sentence it's 0.5, the
    # abs diff is 0.2). The reward is the negative loss.
    # If the highest abs diff among the sentiment scores is lower than epsilon, the goal is reached.
    def custom_rewards(self, generated_sentences, target_sentiment):
        sentiment_outputs = self.apply_sentiment_model(generated_sentences)
        rewards = []
        max_diff = []
        for i,senti in enumerate(sentiment_outputs):
            sum_scores = 0.0
            max_diff_score = None
            for k,score in senti.items():
                diff_score = abs(target_sentiment.get(k) - score)
                sum_scores -= diff_score
                if max_diff_score is None or max_diff_score < diff_score:
                    max_diff_score = diff_score
            rewards.append(sum_scores/len(senti))
            max_diff.append(max_diff_score)
            #print(f"  {i}; {generated_sentences[i]}; {senti}; {rewards[-1]}")
        return (rewards, max_diff)


    # Checks that a generated sentence 'new_part' has the appriate length
    # and satisfies the Levenshtein requirement.
    # Returns the decoded text if ok, None if not.
    def validate_generated_sentence(self, tokenizer, new_part, min_levenshtein_string=None, min_len=40, max_len=60):
        if len(new_part) > 0 and new_part[0] != tokenizer.eos_token_id:
            generated_text = tokenizer.decode(new_part, skip_special_tokens=True)
            if len(generated_text)>=min_len and len(generated_text)<=max_len:
                if min_levenshtein_string is None:
                    return generated_text
                else:
                    if distance(min_levenshtein_string, generated_text)>=30:
                        return generated_text
        return None


    # Runs the GPT2 model to generate a list of `n` sentences based on the given `prompt`.
    def generate_some_sentences(self, model, tokenizer, prompt, n, max_length=15, temperature=0.95, topk=50):
        print(f"Generating {n} sentences... ")
        # Tokenize input prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        input_length = len(input_ids[0])
        generated_sentences = []
        for sample in range(n):
            print(f"\r {sample}/{n} ",end='')
            generated_text = None
            while generated_text is None:
                output = model.generate(input_ids, max_length=input_length+max_length, num_return_sequences=1, do_sample=True, temperature=temperature, top_k=topk, pad_token_id=tokenizer.eos_token_id)
                new_part =output[0, input_length:]
                generated_text = self.validate_generated_sentence(tokenizer, new_part)
                generated_sentences.append({'query': generated_text, 'tokens': new_part.tolist()})
        print()
        #for i,s in enumerate(generated_sentences):
        #    print(f"SENT {i}: {s}")
        return generated_sentences


    def find_close_sentiment(self, max_epochs=9999, max_gen_sent_len=12, verbose = False):

        # Apply to input sentences
        target_sentiment = self.apply_sentiment_model([self.target])[0]

        ds = MyDataset(self.generate_some_sentences(gpt2_model, gpt2_tokenizer, self.target, self.sample_size, max_length=max_gen_sent_len))
        dataloader = torch.utils.data.DataLoader(ds, batch_size=self.ppo_config['batch_size'], collate_fn=collator)

        my_ppo_config = PPOConfig(**self.ppo_config)
        ppo_trainer = PPOTrainer(my_ppo_config, gpt2_model, gpt2_model_ref, gpt2_tokenizer, dataset=ds)

        for epoch in range(max_epochs):
            for batch_id,batch in enumerate(dataloader):
                query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
                text_queries = batch["query"]
                response_tensors = []
                batch['response'] = []
                for i in range(len(query_tensors)):
                    query_len = len(query_tensors[i])
                    generated_text = None
                    while generated_text is None:
                        response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0), max_new_tokens=max_gen_sent_len, **gen_kwargs)
                        generated_text = self.validate_generated_sentence(gpt2_tokenizer, response.squeeze()[query_len:], self.target)
                    response_tensors.append(response.squeeze()[query_len:])
                    batch['response'].append(generated_text)

                #### Compute sentiment score
                rewards_list, max_diff_list = self.custom_rewards(batch['response'], target_sentiment)
                idx_closest_sentiment = np.argmin(max_diff_list)
                result_sentence = batch['response'][idx_closest_sentiment]
                result_sentiment = self.apply_sentiment_model([result_sentence])[0]
                if verbose:
                    print(f"   Target sentiment: {target_sentiment}")
                    print(f"                for sentence: '{input_string}'")
                    print(f"   Current closest sentiment: {result_sentiment}")
                    print(f"                for sentence: '{result_sentence}'")
                # the highest abs diff is used to check that all the scores are close to the target sentiment, since
                # if the highest abs diff is lower than epsilon then we're good.
                if max_diff_list[idx_closest_sentiment] <= self.epsilon:
                    return ProbeResult(result_sentence, result_sentiment)
                rewards = torch.tensor(rewards_list).to(device)
                #### Run PPO step 
                reshaped_rewards = [ torch.tensor(v, requires_grad=True) for v in rewards.clone().detach().requires_grad_(True) ]
                stats = ppo_trainer.step(query_tensors, response_tensors, reshaped_rewards)
     
                print(f"\n***** EPOCH {epoch}, batch {batch_id}: MEAN = {torch.mean(rewards).cpu().numpy()}, CLOSEST HIGHEST DIFF = {max_diff_list[idx_closest_sentiment]}\n")


