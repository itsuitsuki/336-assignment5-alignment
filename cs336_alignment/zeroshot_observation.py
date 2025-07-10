import json
# read the jsonl
with open("cs336_alignment/zeroshot_math_eval.jsonl", "r") as f:
    # 10 cases where format is 0 but answer is 1
    all_data = [json.loads(line) for line in f.readlines()]

format_0_examples = [line for line in all_data if line["reward"]["format_reward"] == 0]
# print(f0a1_examples)
# with format 0 log
with open("cs336_alignment/format_0.log", "w") as f_out:
    for e in format_0_examples[:10]:
        f_out.write(f"Problem: {e['problem']}\n")
        f_out.write(f"Answer: {e['answer']}\n")
        f_out.write(f"Groundtruth: {e['groundtruth']}\n")
        f_out.write(f"Reward: {e['reward']}\n")
        f_out.write("-" * 50 + "\n")
# 10 cases where format is 1 but answer is 0

f1a0_examples = [line for line in all_data if line["reward"]["format_reward"] == 1 and line["reward"]["answer_reward"] == 0]
with open("cs336_alignment/f1a0.log", "w") as f_out:
    for e in f1a0_examples[:10]:
        f_out.write(f"Problem: {e['problem']}\n")
        f_out.write(f"Answer: {e['answer']}\n")
        f_out.write(f"Groundtruth: {e['groundtruth']}\n")
        f_out.write(f"Reward: {e['reward']}\n")
        f_out.write("-" * 50 + "\n")
        
# OBSERVATION
# NOTE: Parser has the responsibility since </think> \n <answer> is never parsed as correct but only </think> (space) <answer>.


