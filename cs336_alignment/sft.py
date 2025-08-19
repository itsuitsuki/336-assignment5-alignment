from cs336_alignment.sft_utils import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step, sft_microbatch_loss_wo_grad
from cs336_alignment.data_utils import preprocess_gsm8k, load_math, load_gsm8k, extract_answer, extract_thinking
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from datasets import load_dataset
from vllm import SamplingParams
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from tqdm import tqdm
from vllm_utils import init_vllm, load_policy_into_vllm_instance

def sft_main(n_epochs=1, bs=128, eval_bs=128, lr=1e-5, grad_accumulation=1., eval_step=200, do_expert_iteration=False):
    """
    Main function to run the SFT process.
    """
    # Initialize wandb
    wandb.init(project="cs336_alignment", name="sft_main",
               config={
                   "n_epochs": n_epochs,
                   "bs": bs,
                   "eval_bs": eval_bs,
                   "lr": lr,
                   "grad_accumulation": grad_accumulation,
                   "eval_step": eval_step,
                   "do_expert_iteration": do_expert_iteration
               }
    )
    # Setup wandb metrics
    wandb.define_metric("train_step")
    # the x‑axis for training
    wandb.define_metric("eval_step")
    # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        trust_remote_code=True,
        max_length=2048,
    )
    print(
        "bos: ", tokenizer.bos_token_id,
        "eos: ", tokenizer.eos_token_id,
        "pad: ", tokenizer.pad_token_id
    )
    # how tf to get them on 2 separate GPUs?
    device_policy = "cuda:1" # Assuming the policy model goes to GPU 1
    policy_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        trust_remote_code=True,
        device_map={"":"cuda:1"}, # if specify on 1 then the device_map should be "cuda:1"
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device_policy)
    eval_model = init_vllm(
        model_id="Qwen/Qwen2.5-Math-1.5B",
        device="cuda:0",  # Assuming the eval model goes to GPU 0
        seed=42,
        gpu_memory_utilization=0.8,
    )
    # try some model things and exit (for debug)
    load_policy_into_vllm_instance(policy_model, eval_model)

    # train_dataset = load_math()["train"]
    # eval_dataset = load_math()["test"]
    
    # load gsm8k
    train_dataset = load_dataset("gsm8k", "main", split="train")
    eval_dataset = load_dataset("gsm8k", "main", split="test")
    # train_dataset = load_gsm8k("train")
    # eval_dataset = load_gsm8k("test")
    print(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")
    # filter out examples with no solution
    train_dataset = preprocess_gsm8k(train_dataset)
    eval_dataset = preprocess_gsm8k(eval_dataset)    
    prompt = open("cs336_alignment/prompts/r1_zero.prompt", "r").read()
    
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)
    # for one example in train_dataset, fill example["problem"] into the prompt batch by batch
    # for a batch of examples i.e. list of dicts
    optimizer.zero_grad()
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=2048, stop=["</answer>"],
        include_stop_str_in_output=True
    )
    cnt = 0
    eval_cnt = 0
    
    if do_expert_iteration:
        print("Doing the initial expert iteration...")
        train_dataset = filter_by_expert_iteration(train_dataset, eval_model, 
                                                   prompt=prompt, n_sample_responses=8, temperature=0.85, top_p=0.8,
                                                   reward_fn=r1_zero_reward_fn, bs=eval_bs)
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")
        print("="*50)
        policy_model.train()
        for i, batch in enumerate(train_dataset.batch(batch_size=bs)):
            torch.cuda.empty_cache()
            # print(f"Policy Device VRAM Usage: {torch.cuda.memory_allocated(device_policy) / 1024**2} MB")
            problem_list = batch["problem"] # list[str]
            solution_list = batch["resp_sol"] if do_expert_iteration else batch["solution"]
            prompt_list = [prompt.format(question=problem) for problem in problem_list]
            # print(prompt_list)
            # print(solution_list)
            input_ids, labels, response_mask = tokenize_prompt_and_output(
                prompt_strs=prompt_list,
                output_strs=solution_list,
                tokenizer=tokenizer
            ).values()
            # print back the encoded-decoded input_ids
            # print(input_ids)
            # print(labels)
            # print(tokenizer.batch_decode(input_ids))
            input_ids = input_ids.to(device_policy)
            labels = labels.to(device_policy)
            response_mask = response_mask.to(device_policy)
            # get the response log probs and token entropy
            # response_log_probs: (B, L, V), token_entropy: (B, L)
            # where B is batch size, L is sequence length, V is vocabulary size
            # response_mask: (B, L) where 1 for response tokens and 0 for prompt or padding tokens
            # print(f"--- Before forward pass (Batch {i}) ---")
            # print(torch.cuda.memory_summary(device=device_policy))
            response_log_probs, token_entropy = get_response_log_probs(
                model=policy_model, 
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True
            ).values()
            # print(f"--- Before backward pass (Batch {i}) ---")
            # print(torch.cuda.memory_summary(device=device_policy))
            n_response_tokens = response_mask.sum()
            loss, metadata = sft_microbatch_train_step(
                response_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=grad_accumulation,
                normalize_constant=n_response_tokens/bs, # token-level
            )
            
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}, Entropy: {token_entropy.mean().item()}")
            # grad clip
            orig_grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                max_norm=1.0,
            )
            # log the response_log_probs and token_entropy
            wandb.log({
                "train/token_entropy": token_entropy.mean().item(),
                "train/loss": loss.item(),
                "train/grad_norm": orig_grad_norm.item(),
                "train_step": cnt,
            })
            del input_ids, labels, response_mask, response_log_probs, token_entropy
            torch.cuda.empty_cache()
            cnt += 1
            
            if cnt % grad_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
            # print(f"Policy Device VRAM Usage: {torch.cuda.memory_allocated(device_policy) / 1024**2} MB")
            # print('-'*50)
            if cnt % eval_step == 0:
                policy_model.eval()
                load_policy_into_vllm_instance(policy_model, eval_model)
                # do eval from eval_dataset
                # eval_losses = []
                eval_ans_true = 0
                eval_all_true = 0
                eval_total = 0
                # eval_model.eval()
                
                with torch.no_grad():
                    # for eval_batch in eval_dataset.batch(batch_size=eval_bs):
                    pbar = tqdm(eval_dataset.batch(batch_size=eval_bs), desc=f"Evaluating Epoch {epoch}")
                    # table = wandb.Table(columns=["question", "whole_generation", "thinking", "answer", "answer_reward", "all_reward"])
                    for eval_batch in pbar:
                        eval_problem_list = eval_batch["problem"]
                        eval_solution_list = eval_batch["solution"]
                        eval_prompt_list = [prompt.format(question=problem) for problem in eval_problem_list]
                        # input_ids, labels, response_mask = tokenizer_prompt_and_output(
                        #     prompt_strs=eval_prompt_list,
                        #     output_strs=eval_solution_list,
                        #     tokenizer=tokenizer
                        # ).values()
                        # # to device
                        # input_ids = input_ids.to(device_policy)
                        # labels = labels.to(device_policy)
                        # response_mask = response_mask.to(device_policy)
                        
                        # response_log_probs = get_response_log_probs(
                        #     model=policy_model,
                        #     input_ids=input_ids,
                        #     labels=labels,
                        #     return_token_entropy=False
                        # )["log_probs"]
                        # loss = sft_microbatch_loss_wo_grad(
                        #     response_log_probs,
                        #     response_mask=response_mask,
                        #     normalize_constant=1
                        # )
                        # eval_losses.append(loss.item())
                        # calculate accuracy by generations from eval_model from eval_prompt_list
                        generations = eval_model.generate(
                            prompts=eval_prompt_list,
                            sampling_params=sampling_params,
                            use_tqdm=False
                        )
                        answers = [gen.outputs[0].text for gen in generations]
                        tmp_rewards = [r1_zero_reward_fn(answer, groundtruth.split("<answer>")[-1].split("</answer>")[0]) for answer, groundtruth in zip(answers, eval_solution_list)]
                        # "reward"
                        eval_total += len(tmp_rewards)
                        eval_ans_true += sum(1 for r in tmp_rewards if r["answer_reward"] == 1)
                        eval_all_true += sum(1 for r in tmp_rewards if r["reward"] == 1)
                        # print(f"Epoch {epoch}, Ans True: {eval_ans_true}, All True: {eval_all_true}, Total: {eval_total}")
                        pbar.set_postfix_str(
                            f"Ans True: {eval_ans_true}, All True: {eval_all_true}, Total: {eval_total}, Ans Accuracy: {eval_ans_true / eval_total if eval_total > 0 else 0}, All Accuracy: {eval_all_true / eval_total if eval_total > 0 else 0}"
                        )
                        # print batch[0] generations
                        
                        # Log generation as a wandb Table (stores text cleanly rather than printing to stdout)
                        
                        # table.add_data(
                        #     eval_problem_list[0],
                        #     eval_problem_list[0] + answers[0],
                        #     extract_thinking(eval_problem_list[0] + answers[0]),
                        #     extract_answer(eval_problem_list[0] + answers[0]),
                        #     int(tmp_rewards[0]["answer_reward"]),
                        #     int(tmp_rewards[0]["reward"]),
                        # )
                    # wandb.log({
                    #     "eval/generations": table,
                    #     "eval_step": eval_cnt,
                    # })

                eval_ans_accuracy = eval_ans_true / eval_total
                eval_all_accuracy = eval_all_true / eval_total
                wandb.log({
                    # "eval/loss": sum(eval_losses) / len(eval_losses),
                    "eval/ans_accuracy": eval_ans_accuracy,
                    "eval/all_accuracy": eval_all_accuracy,
                    "eval_step": eval_cnt,
                })
                policy_model.train()
                
                # print(f"Eval Loss: {sum(eval_losses) / len(eval_losses)}, Ans Accuracy: {eval_ans_accuracy}, All Accuracy: {eval_all_accuracy}")
        if do_expert_iteration:
            train_dataset = filter_by_expert_iteration(train_dataset, eval_model, prompt, n_sample_responses=8, temperature=0.85, top_p=0.8, reward_fn=r1_zero_reward_fn, bs=eval_bs)
    # end eval model
    del eval_model

def filter_by_expert_iteration(dataset, model, prompt: str, n_sample_responses=16, temperature=0.85, top_p=0.95, reward_fn=r1_zero_reward_fn, bs=32):
    new_dataset = []
    sampling_params = SamplingParams(
        n=n_sample_responses, temperature=temperature, top_p=top_p, max_tokens=2048, stop=["</answer>"], min_tokens=4, seed=42,
        include_stop_str_in_output=True
    )
    for batch in tqdm(dataset.batch(batch_size=bs//n_sample_responses), desc="Filtering & Synthesizing by expert iteration"):
        problem_list = batch["problem"]
        solution_list = batch["solution"] # use the original thing
        answer_list = [extract_answer(solution) for solution in solution_list]
        prompt_list = [prompt.format(question=problem) for problem in problem_list]

        responses = model.generate(prompt_list, sampling_params, use_tqdm=False)
        for i, response in enumerate(responses):
            # multiple completion outputs in response
            for output in response.outputs:
                if reward_fn(output.text, answer_list[i])["reward"] == 1:
                    # make the answer boxed. now it's like output.text=...</think> <answer>...</answer> --> ...</think> <answer>\\boxed{...}</answer>
                    new_dataset.append({
                        "problem": problem_list[i],
                        "solution": solution_list[i],
                        "resp_sol": output.text
                    })
    from datasets import Dataset
    
    new_dataset = Dataset.from_list(new_dataset)
    return new_dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--bs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accumulation", type=int, default=1)
    parser.add_argument("--eval_step", type=int, default=500)
    parser.add_argument("--eval_bs", type=int, default=256)
    parser.add_argument("--do_expert_iteration", action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    sft_main(n_epochs=args.n_epochs, bs=args.bs, eval_bs=args.eval_bs, lr=args.lr, grad_accumulation=args.grad_accumulation, eval_step=args.eval_step, do_expert_iteration=args.do_expert_iteration)
    # Note: Adjust the parameters as needed for your training setup.