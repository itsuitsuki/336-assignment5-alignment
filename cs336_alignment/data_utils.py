import json
import re 
from datasets import load_dataset

def load_gsm8k(split):
    # read ./data/gsm8k/<split>.jsonl
    with open(f"./data/gsm8k/{split}.jsonl", "r") as f:
        dataset = [json.loads(line) for line in f]
    return dataset

def filter_gsm8k_to_think_answer_format(
    gsm8k_example,
):
    # find the last "####" pattern and change to </think> <answer>
    # problem = gsm8k_example["question"]
    orig_ans_str = gsm8k_example["answer"]
    
    # replace the last occurrence of "####" with "</think> <answer>"
    # the last "####" is in "#### 42abc" where 42abc is the answer and the pattern is the end of the answer.
    # add boxed 
    answer = re.sub(r"####\s*(.*)", r"</think> <answer>\1</answer>", orig_ans_str, count=1)
    answer += " \n"
    # no need to add <think> at the beginning, since the prompt ends with <think>
    return {
        "problem": gsm8k_example["question"],
        "solution": answer,
    }

def preprocess_gsm8k(
    dataset,
    shuffle: bool = False,
    seed = 42,
):
    # filter w/ only w/ a sol
    dataset = dataset.filter(lambda x: x["answer"] is not None)
    # map to the think-answer format
    dataset = dataset.map(filter_gsm8k_to_think_answer_format, remove_columns=["question", "answer"], load_from_cache_file=True)
    # shuffle?
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    return dataset

def load_math():
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", streaming=False) # "train" and "test"
    return ds