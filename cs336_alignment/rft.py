from data_utils import preprocess_gsm8k
from vllm_utils import init_vllm, load_policy_into_vllm_instance
from vllm import SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from cs336_alignment.rft_utils import grpo_microbatch_train_step, compute_group_normalized_rewards
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import wandb
from tqdm import tqdm

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def rft_grpo_gsm8k(args):
    wandb.init(project="cs336_alignment", 
               name=f"grpo_rft_gsm8k_{args.loss_type}",
               config=vars(args))
    wandb.define_metric("train_step") # microbatch step
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, max_length=args.max_len)
    device_policy = "cuda:1"
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map={"": "cuda:1"}, # if specify on 1 then the device_map should be "cuda:1"
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device_policy)
    eval_model = init_vllm(args.model_name_or_path, device="cuda:0", seed=args.seed, gpu_memory_utilization=0.85)

    train_dataset = load_dataset("gsm8k", "main", split="train")
    eval_dataset = load_dataset("gsm8k", "main", split="test")
    print(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")
    train_dataset = preprocess_gsm8k(train_dataset)
    eval_dataset = preprocess_gsm8k(eval_dataset)
    prompt = open("cs336_alignment/prompts/r1_zero.prompt").read()
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    optimizer.zero_grad()
    train_sampling_params = SamplingParams(
        n=args.group_size,
        temperature=args.sampling_temperature,
        top_p=args.sampling_top_p,
        max_tokens=args.max_len,
        min_tokens=4,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    eval_sampling_params = SamplingParams(
        max_tokens=args.max_len,
        min_tokens=4,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    cnt = 0 # NOTE: THE MICROBATCH CNT, ALSO FOR TRAINING STEPS
    train_batch_size = args.train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    group_size = args.group_size
    rollout_batch_size = args.rollout_batch_size
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size # should divide by group size??
    print("="*50)
    policy_model.train()
    load_policy_into_vllm_instance(policy_model, eval_model)
    # training_table = wandb.Table(
    #     columns = ["rollout_step", "group_id", "idx", "problem", "thinking", "answer", "groundtruth_thinking", "groundtruth_answer", "raw_rewards", "adv", "mean_r", "std_r"], log_mode="INCREMENTAL"
    # )
    # eval_table = wandb.Table(
    #     columns = ["rollout_step", "idx", "problem", "thinking", "answer", "groundtruth_thinking", "groundtruth_answer", "raw_rewards", "adv"], log_mode="INCREMENTAL"
    # )
    for grpo_step in range(args.n_grpo_steps):
        # every step, sample a rollout batch. the training is decoupled! so the training batch thing does nothing w/ this section
        policy_model.train()
        with torch.no_grad():
            batch = train_dataset.shuffle(seed=grpo_step).select(range(n_prompts_per_rollout_batch))
            torch.cuda.empty_cache()
            problem_list = batch["problem"] # list[str]
            solution_list = batch["solution"]
            repeated_prbls = [prb for prb in problem_list for _ in range(args.group_size)]
            # repeated_solutions = [sol for sol in solution_list for _ in range(args.group_size)]
            repeated_grths = [sol.split("<answer>")[-1].split("</answer>")[0] for sol in solution_list for _ in range(args.group_size)]
            prompt_list = [prompt.format(question=problem) for problem in problem_list]
            repeated_prompts = [pr for pr in prompt_list for _ in range(args.group_size)]
            # use eval model as old model to generate args.group_size generations per prompt
            generations = eval_model.generate(
                prompt_list,
                train_sampling_params,
                use_tqdm=False)
            generations = [gen.outputs[j].text for gen in generations for j in range(args.group_size)]
            adv, raw_rewards, metadata = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn,
                rollout_responses=generations,
                repeated_ground_truths=repeated_grths,
                group_size=args.group_size,
                advantage_eps=1e-6,
                normalize_by_std=(not args.drgrpo)
            )
            effective_groups = []
            effective_idxs = []
            print('='*50)
            print(f"GRPO Step {grpo_step} Before-training Accuracy: {raw_rewards.mean().item()}")
            for group_id in range(n_prompts_per_rollout_batch):
                print('='*50)
                # if adv is not all zero (reward is not all the same), record the effective indices
                if metadata["std"][group_id] == 0:
                    # FIXME: NOT DEBUGGED, DAPO RE-SAMPLING (DYNAMIC SAMPLING)
                    for retry in range(args.n_dapo_resample_retries):
                        print(f"Retry {retry + 1}/{args.n_dapo_resample_retries} for group {group_id}")
                        # resample 1 prompt from the train dataset
                        sample = train_dataset.shuffle(grpo_step).select([n_prompts_per_rollout_batch - 1 + group_id * args.n_dapo_resample_retries + retry])
                        tmp_problem = sample["problem"]
                        tmp_solution = sample["solution"]
                        tmp_prompt_list = [prompt.format(question=tmp_problem[0])]
                        tmp_gens = eval_model.generate(
                            tmp_prompt_list,
                            train_sampling_params,
                            use_tqdm=False
                        )
                        tmp_gens = [gen.outputs[j].text for gen in tmp_gens for j in range(args.group_size)]
                        tmp_adv, tmp_raw_rewards, tmp_metadata = compute_group_normalized_rewards(
                            reward_fn=r1_zero_reward_fn,
                            rollout_responses=tmp_gens,
                            repeated_ground_truths=[tmp_solution[0]] * args.group_size,
                            group_size=args.group_size,
                            advantage_eps=1e-6,
                            normalize_by_std=(not args.drgrpo)
                        )
                        if tmp_metadata["std"][0] != 0:
                            # success, start to substitute
                            problem_list[group_id] = tmp_problem[0]
                            solution_list[group_id] = tmp_solution[0]
                            prompt_list[group_id] = tmp_prompt_list[0]
                            repeated_prbls[group_id * args.group_size : (group_id + 1) * args.group_size] = [tmp_problem[0]] * args.group_size
                            repeated_grths[group_id * args.group_size : (group_id + 1) * args.group_size] = [tmp_solution[0]] * args.group_size
                            generations[group_id * args.group_size : (group_id + 1) * args.group_size] = tmp_gens
                            adv[group_id * args.group_size : (group_id + 1) * args.group_size] = tmp_adv
                            raw_rewards[group_id * args.group_size : (group_id + 1) * args.group_size] = tmp_raw_rewards
                            metadata["mean"][group_id] = tmp_metadata["mean"][0]
                            metadata["std"][group_id] = tmp_metadata["std"][0]
                            metadata["meta_rewards"][group_id * args.group_size : (group_id + 1) * args.group_size] = tmp_metadata["meta_rewards"]
                            print(f"Group {group_id} Resampled Successfully")
                            break
                        else:
                            print(f"Group {group_id} Resampling Failed")
                if metadata["std"][group_id] != 0:
                    effective_groups.append(group_id)
                    effective_idxs.extend(list(range(group_id * args.group_size, (group_id + 1) * args.group_size)))
                print(f"Group {group_id}\nProblem: {problem_list[group_id]}\nMean: {metadata['mean'][group_id]}, STD: {metadata['std'][group_id]}")
                print(f"Rewards: {raw_rewards[group_id * args.group_size : (group_id + 1) * args.group_size].tolist()}")
                print(f"Format Rewards: {[fr['format_reward'] for fr in metadata['meta_rewards'][group_id * args.group_size : (group_id + 1) * args.group_size]]}")
                print(f"Advantage: {adv[group_id * args.group_size : (group_id + 1) * args.group_size].tolist()}")
        # training!
        print('='*50)
        if effective_groups:
            print(f"Effective Groups: {effective_groups}, #: {len(effective_groups)}")
        else:
            print("No effective groups found. Training failed.")
            return
        tokenizer_out = tokenizer(
            repeated_prompts,
            generations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        ids = tokenizer_out.input_ids # torch tensor
        prompt_lens = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in repeated_prbls]
        response_mask = tokenizer_out.attention_mask # torch tensor
        for i, prompt_len in enumerate(prompt_lens):
            response_mask[i, :prompt_len] = 0
        for iteration in range(args.n_iterations):
            old_log_probs = torch.zeros_like(ids[:, 1:], dtype=torch.float32).to(device_policy)
            # is_clipped = torch.zeros_like(ids[:, 1:], dtype=torch.float32).to(device_policy)
            print('='*50)
            for microbatch_step in tqdm(range(n_microbatches_per_rollout_batch), desc=f"Microbatch Steps in iteration {iteration}, GRPO Step {grpo_step}"):
                # if no idxs are in effective_idxs, continue
                if set(effective_idxs).isdisjoint(set(range(microbatch_step * micro_train_batch_size, (microbatch_step + 1) * micro_train_batch_size))):
                    cnt += 1
                    wandb.log(
                        {
                            "train/policy_loss": float('nan'),
                            "train/orig_grad_norm": float('nan'),
                            "train/mean_microbatch_entropy": float('nan'),
                            "train/clipped_fraction": float('nan'),
                            # "train/train_mean_raw_rewards": raw_rewards[
                            #     microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                            # ].detach().cpu().mean().item(),
                            # "train/train_mean_adv": adv[
                            #     microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                            # ].detach().cpu().mean().item(),
                            "train_step": cnt
                        }
                    )
                    continue
                torch.cuda.empty_cache()
                # get policy log probs
                # logits is for the next token
                tmp_ids = ids[
                            microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                        ].to(device_policy)
                logits = policy_model.forward(tmp_ids, output_hidden_states=False).logits # (bs, seqlen+1, vocab_size)
                logits = logits[:, :-1, :] # (bs, seqlen, vocab_size)
                tmp_ids = tmp_ids[:, 1:]
                # we need to shift
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1) # (bs, seqlen, vocab_size)
                
                with torch.no_grad():
                    entropy = -torch.sum(log_probs * torch.exp(log_probs), dim=-1) # (bs, seqlen)
                log_probs = log_probs.gather(-1, tmp_ids.unsqueeze(-1)).squeeze(-1) # (bs, seqlen-1)
                del logits
                torch.cuda.empty_cache()
                with torch.no_grad():
                    if iteration == 0:
                        old_log_probs_micro = log_probs.detach()
                    else:
                        old_log_probs_micro = old_log_probs[
                            microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                        ].detach()

                tmp_response_mask = response_mask[
                    microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                ][:, 1:].to(device_policy)
                pg_loss, metadata_tmp = grpo_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=tmp_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=args.loss_type,
                    raw_rewards=raw_rewards[
                        microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                    ],
                    advantages=adv[
                        microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                    ],
                    old_log_probs=old_log_probs_micro,
                    cliprange=args.cliprange,
                    normalize_method=args.loss_normalize_method,
                )
                cnt += 1
                orig_grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                if cnt % gradient_accumulation_steps == 0:
                    optimizer.step()
                    # lr_scheduler.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    old_log_probs[
                        microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                    ] = log_probs.detach()
                    # is_clipped[
                    #     microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                    # ] = metadata_tmp["is_clipped"]
                    # print('-'*50)
                    # print(f"Microbatch {microbatch_step}, Step {cnt}, Policy Loss: {pg_loss.item()}, Orig Grad Norm: {orig_grad_norm.item()}, Mean Microbatch Entropy: {entropy.mean().item()}")
                    # print(f"Microbatch {microbatch_step}, Step {cnt}, Clipped Tokens: {metadata_tmp['clip_fraction']}")
                    # print(f"Microbatch {microbatch_step}, Step {cnt}, Mean Reward: {raw_rewards[
                    #     microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                    # ].detach().cpu().mean().item()}, Mean Advantage: {adv[
                    #     microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                    # ].detach().cpu().mean().item()}")
                    wandb.log(
                        {
                            "train/policy_loss": pg_loss.item(),
                            "train/orig_grad_norm": orig_grad_norm.item(),
                            "train/mean_microbatch_entropy": entropy.mean().item(),
                            "train/clipped_fraction": metadata_tmp["is_clipped"].float().mean().item(),
                            # "train/train_mean_raw_rewards": raw_rewards[
                            #     microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                            # ].detach().cpu().mean().item(),
                            # "train/train_mean_adv": adv[
                            #     microbatch_step * micro_train_batch_size : (microbatch_step + 1) * micro_train_batch_size
                            # ].detach().cpu().mean().item(),
                            "train_step": cnt
                        }
                    )
                    
        # eval
        policy_model.eval()
        load_policy_into_vllm_instance(policy_model, eval_model)
        eval_format_true = 0
        eval_all_true = 0
        eval_total = 0
        # eval_model.eval()
        
        with torch.no_grad():
            # for eval_batch in eval_dataset.batch(batch_size=eval_bs):
            pbar = tqdm(eval_dataset.batch(batch_size=args.eval_batch_size), desc=f"Evaluating GRPO Step {grpo_step}")
            for eval_batch in pbar:
                eval_problem_list = eval_batch["problem"]
                eval_solution_list = eval_batch["solution"]
                eval_prompt_list = [prompt.format(question=problem) for problem in eval_problem_list]
                generations = eval_model.generate(
                    prompts=eval_prompt_list,
                    sampling_params=eval_sampling_params,
                    use_tqdm=False
                )
                generations = [gen.outputs[0].text for gen in generations]
                tmp_rewards = [r1_zero_reward_fn(gen, groundtruth.split("<answer>")[-1].split("</answer>")[0]) for gen, groundtruth in zip(generations, eval_solution_list)]
                # "reward"
                eval_total += len(tmp_rewards)
                eval_format_true += sum(1 for r in tmp_rewards if r["format_reward"] == 1)
                eval_all_true += sum(1 for r in tmp_rewards if r["reward"] == 1)
                pbar.set_postfix_str(
                    f"Format True: {eval_format_true}, All True: {eval_all_true}, Total: {eval_total}, Format Accuracy: {eval_format_true / eval_total if eval_total > 0 else 0}, All Accuracy: {eval_all_true / eval_total if eval_total > 0 else 0}"
                )

        eval_format_accuracy = eval_format_true / eval_total
        eval_all_accuracy = eval_all_true / eval_total
        print("="*50)
        print("Evaluation")
        print(
            f"GRPO Step {grpo_step}, Format Accuracy: {eval_format_accuracy}, All Accuracy: {eval_all_accuracy}"
        )
        wandb.log({
            "eval/format_accuracy": eval_format_accuracy,
            "eval/all_accuracy": eval_all_accuracy,
            "eval_step": grpo_step,
        })

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_type", type=str, choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"], default="grpo_clip")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_top_p", type=float, default=0.9)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--n_iterations", type=int, default=1, help="aka. epochs_per_rollout_batch; number of training iterations per rollout batch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--cliprange", type=float, default=0.2, help="clipping range for grpo_clip")
    parser.add_argument("--drgrpo", type=bool, default=False, help="whether to use dynamic rollout for grpo i.e. not to use normalizing by std")
    parser.add_argument("--loss_normalize_method", type=str, default="divide_unmasked_len", choices=["divide_unmasked_len", "divide_max_len"], help="method for normalizing loss")
    parser.add_argument("--n_dapo_resample_retries", type=int, default=3, help="number of retries for DAPO resampling")
    args = parser.parse_args()
    seed_everything(args.seed)
    rft_grpo_gsm8k(args)
    
if __name__ == '__main__':
    main()