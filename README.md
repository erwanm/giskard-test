# Erwan's solution to the technical exercise for the position of ML researcher @ Giskard
(Original instructions below)

## Part 1

My solution relies on the [TRL (Transformer Reinforcement Learning) module](https://huggingface.co/docs/trl/index) to tune a GPT2 model into generating sentences for which the sentiment is as close as possible to the target sentence. A dataset of sentences is generated by GPT2 initially using the target sentence as prompt. Then, at every epoch, these sentences are used as prompts for the progressively tuned GPT2 model to generate new sentences. A custom reward function compares the sentiment of these sentences to the sentiment of the target sentence, rewarding the closest ones. Eventually the model specializes into generating sentences with a sentiment close to the target, and the process stops when a sentence satisfying the criteria is found. 

prompts are made of sentences themselves generated by GPT2 

- Strongly inspired from the notebook at [https://huggingface.co/docs/trl/v0.1.1/en/sentiment_tuning](https://huggingface.co/docs/trl/v0.1.1/en/sentiment_tuning)
- Limitation: English-only. Replacing GPT2 with a multilingual model should allow the process to accept a multilingual input.


## Part 2 

See my review here: [paper-review.md](https://github.com/erwanm/giskard-test/blob/main/paper-review.md)
  
# Technical exercise - ML researcher @ Giskard

As part of our recruitment process, we kindly ask you to complete the following
exercise in 10 days.

The exercise is divided into two parts:
- Part 1 is a practical, applied research & coding exercise
- Part 2 is a paper review

This repository provides a template to submit your assignment. You can clone it,
complete it with your solutions and submit it by sharing it privately to
rabah@giskard.ai (`rabah-khalek` on Github) and matteo@giskard.ai (`mattbit` on
Github).


## Part 1: Sentiment Analysis Challenge

We would like you develop an algorithm that, given a sentiment analysis model
and a target sentence, identifies a different sentence which has same sentiment
score when rounded to _n_ decimal digits.

For example, given the model https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student,
the target sentence “My grandmother's secret sauce is the best ever made!”, and
an objective of `3` decimal digits, your algorithm may find a sentence such as:

> I feel it should be a positive thing for us to look

As you can see, the sentiment scores of the two sentences rounded up to **`3`**
decimal digits is equal:

| Sentence | Positive score | Neutral score | Negative score |
| --- | --- | --- | --- |
| My grandmother's secret sauce is the best ever made! | 0.9619 | 0.0256 | 0.0123 |
| I feel it should be a positive thing for us to look | 0.9615 | 0.0262 | 0.0121 |


**Guidelines:**

- Ensure a minimum [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) of **`30`** between the target sentence and the sentence found by your algorithm.
- Keep the sentence within the length range of **`40`** to **`60`** characters.
- Ensure your algorithm works well when matching scores rounded to **`5`**
  decimal digits.
- Implement this algorithm in the fictitious `chameleon` Python package
  provided as part of this repository (see docs [chameleon.md](chameleon.md)).


## Part 2: Paper Review

We would like you to read the following paper
[arXiv:2310.18344](https://arxiv.org/abs/2310.18344) and write a short review.
The format should be similar to a peer review, but addressed at your colleagues
(no need for excessively formal language).

**Guidelines:**

- Write your review in `paper-review.md` file, no need to dedicate time to styling
- Clear, straight-to-the-point language is preferred
- Limit the review to approximately 500 words (equivalent to one page)
