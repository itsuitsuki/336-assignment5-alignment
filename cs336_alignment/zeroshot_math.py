from datasets import load_dataset
import torch
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn
from vllm.distributed.parallel_state import destroy_model_parallel
from typing import Callable

def load_math():
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", streaming=False)
    return ds

def generate_ans(llm, sampling_params, prompt):
    responses = llm.generate(
        prompt,
        sampling_params=sampling_params,
    )
    answers: list = [response.outputs[0].text for response in responses]
    return answers


def demo():
    ds = load_math()["test"]
    print("Example from the dataset:")
    random_example = next(iter(ds))
    print(random_example)

    # read from ./prompts/r1_zero.prompt
    prompt = open("cs336_alignment/prompts/r1_zero.prompt", "r").read()
    print("Prompt:")
    print(prompt)
    
    # qwen 2.5 math 1.5b
    llm = LLM(
        model="Qwen/Qwen2.5-Math-1.5B",
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=4096, stop=["</answer>"],
        include_stop_str_in_output=True
    )
    

    model_input = prompt.format(question=random_example["problem"])
    print("Input:")
    print(model_input)
    print("Generating answer...")
    answer = generate_ans(llm, sampling_params, model_input)[0]
    groundtruth = random_example["solution"]
    reward = r1_zero_reward_fn(answer, groundtruth)
    print("Response:")
    print(answer)
    print("Groundtruth:")
    print(groundtruth)
    print("Reward:")
    print(reward)
    
    print("Done.")

    del llm
    destroy_model_parallel()
    del ds
    torch.cuda.empty_cache()

def evaluate_math(
                    vllm_model: LLM, 
                    reward_fn: Callable[[str, str], dict[str, float]],
                    eval_sampling_params: SamplingParams,
                    train_test="test", # "train" or "test"
                    batch_size=32,
                ):
    ds = load_math()[train_test]

    # serialize example, model generation and reward computation to a jsonl file
    import json
    with open("cs336_alignment/zeroshot_math_eval.jsonl", "w") as f:
        rewards = []
        for batch in ds.batch(batch_size):
            problem_list = batch["problem"]
            
            # model_input = open("cs336_alignment/prompts/r1_zero.prompt", "r").read().format(question=example["problem"])
            model_inputs = [open("cs336_alignment/prompts/r1_zero.prompt", "r").read().format(question=problem) for problem in problem_list]
            answers = generate_ans(vllm_model, eval_sampling_params, model_inputs) # type: list[str]

            solution_list = batch["solution"] # groundtruth
            tmp_rewards = [reward_fn(answer, groundtruth) for answer, groundtruth in zip(answers, solution_list)]  # type: list[dict[str, float]]
            # write
            for problem, answer, groundtruth, reward in zip(problem_list, answers, solution_list, tmp_rewards):
                f.write(json.dumps({
                    "problem": problem,
                    "answer": answer,
                    "groundtruth": groundtruth,
                    "reward": reward,
                }) + "\n")
            rewards.extend(tmp_rewards)

    # rewards is a list of dicts, each dict contains the reward for each example
    sum_format_reward = sum([reward["format_reward"] for reward in rewards])
    sum_answer_reward = sum([reward["answer_reward"] for reward in rewards])
    sum_reward = sum([reward["reward"] for reward in rewards])
    print("-" * 50)
    print(f"Format Correctness #: {sum_format_reward:.4f}")
    print(f"Answer Correctness #: {sum_answer_reward:.4f}")
    print(f"Total Reward #: {sum_reward:.4f}")
    print(f"Total Examples #: {len(rewards)}")
    print("-" * 50)
    print(f"Average Format Reward: {sum_format_reward / len(rewards):.4f}")
    print(f"Average Answer Reward: {sum_answer_reward / len(rewards):.4f}")
    print(f"Average Total Reward: {sum_reward / len(rewards):.4f}")
    print("-" * 50)
    
    return rewards

def main():
    # llm initialization
    llm = LLM(
        model="Qwen/Qwen2.5-Math-1.5B",
        trust_remote_code=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=4096, stop=["</answer>"],
        include_stop_str_in_output=True
    )
    # evaluate the model on the math dataset
    evaluate_math(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        eval_sampling_params=sampling_params,
        train_test="test",  # or "train"
        batch_size=128
    )
    
if __name__ == "__main__":
    main()