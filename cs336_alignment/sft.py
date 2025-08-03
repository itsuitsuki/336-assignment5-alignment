from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
from unittest.mock import patch
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
import wandb
from datasets import load_dataset


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def sft_main(n_epochs=1, bs=128, lr=1e-5, grad_accumulation=1, eval_step=200):
    """
    Main function to run the SFT process.
    """
    # Initialize wandb
    wandb.init(project="cs336_alignment", name="sft_main")
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
        max_length=4096,
    )
    # how tf to get them on 2 separate GPUs?
    device_policy = "cuda:1" # Assuming the policy model goes to GPU 1
    policy_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        trust_remote_code=True,
        device_map={"":"cuda:1"}, # if specify on 1 then the device_map should be "cuda:1"
        # fp16
        torch_dtype=torch.bfloat16,
    ).to(device_policy)
    eval_model = init_vllm(
        model_id="Qwen/Qwen2.5-Math-1.5B",
        device="cuda:0",  # Assuming the eval model goes to GPU 0
        seed=42,
        gpu_memory_utilization=0.8,
    )
    # try some model things and exit (for debug)
    load_policy_into_vllm_instance(policy_model, eval_model)
    print("Policy model loaded into vLLM instance.")
    from cs336_alignment.sft_utils import tokenizer_prompt_and_output, get_response_log_probs, sft_microbatch_train_step, sft_microbatch_loss_wo_grad
    from cs336_alignment.sft_utils import preprocess_gsm8k_dataset, load_math
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    train_dataset = load_math()["train"]
    eval_dataset = load_math()["test"]
    
    # # load gsm8k
    # train_dataset = load_dataset("gsm8k", split="train")
    # eval_dataset = load_dataset("gsm8k", split="test")
    
    # # filter out examples with no solution
    # train_dataset = preprocess_gsm8k_dataset(train_dataset)
    # eval_dataset = preprocess_gsm8k_dataset(eval_dataset)
    
    # print the size of the datasets
    print(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")
    
    
    prompt = open("cs336_alignment/prompts/r1_zero.prompt", "r").read()
    
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)
    # for one example in train_dataset, fill example["problem"] into the prompt batch by batch
    # for a batch of examples i.e. list of dicts
    optimizer.zero_grad()

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=4096, stop=["</answer>"],
        include_stop_str_in_output=True
    )
    cnt = 0
    policy_model.train()
    

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        policy_model.train()
        for i, batch in enumerate(train_dataset.batch(batch_size=bs)):
            cnt += 1
            problem_list = batch["problem"] # list[str]
            solution_list = batch["solution"]
            prompt_list = [prompt.format(question=problem) for problem in problem_list]
            input_ids, labels, response_mask = tokenizer_prompt_and_output(
                prompt_strs=prompt_list,
                output_strs=solution_list,
                tokenizer=tokenizer
            ).values()
            input_ids = input_ids.to(device_policy)
            labels = labels.to(device_policy)
            response_mask = response_mask.to(device_policy)
            # get the response log probs and token entropy
            # response_log_probs: (B, L, V), token_entropy: (B, L)
            # where B is batch size, L is sequence length, V is vocabulary size
            # response_mask: (B, L) where 1 for response tokens and 0 for prompt or padding tokens
            response_log_probs, token_entropy = get_response_log_probs(
                model=policy_model, 
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True
            ).values()
            loss, metadata = sft_microbatch_train_step(
                response_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=grad_accumulation,
                normalize_constant=1
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
            })
            # wandb table log generations from problem_list
            
            
            if cnt % grad_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if cnt % eval_step == 0:
                policy_model.eval()
                load_policy_into_vllm_instance(policy_model, eval_model)
                # do eval from eval_dataset
                eval_losses = []
                eval_ans_true = 0
                eval_all_true = 0
                eval_total = 0
                # eval_model.eval()
                
                with torch.no_grad():
                    for eval_batch in eval_dataset.batch(batch_size=bs):
                        eval_problem_list = eval_batch["problem"]
                        eval_solution_list = eval_batch["solution"]
                        eval_prompt_list = [prompt.format(question=problem) for problem in eval_problem_list]
                        input_ids, labels, response_mask = tokenizer_prompt_and_output(
                            prompt_strs=eval_prompt_list,
                            output_strs=eval_solution_list,
                            tokenizer=tokenizer
                        ).values()
                        # to device
                        input_ids = input_ids.to(device_policy)
                        labels = labels.to(device_policy)
                        response_mask = response_mask.to(device_policy)
                        
                        response_log_probs = get_response_log_probs(
                            model=policy_model,
                            input_ids=input_ids,
                            labels=labels,
                            return_token_entropy=False
                        )["log_probs"]
                        loss = sft_microbatch_loss_wo_grad(
                            response_log_probs,
                            response_mask=response_mask,
                            normalize_constant=1
                        )
                        eval_losses.append(loss.item())
                        # calculate accuracy by generations from eval_model from eval_prompt_list
                        generations = eval_model.generate(
                            prompts=eval_prompt_list,
                            sampling_params=sampling_params,
                            use_tqdm=False
                        )
                        answers = [gen.outputs[0].text for gen in generations]
                        tmp_rewards = [r1_zero_reward_fn(answer, groundtruth) for answer, groundtruth in zip(answers, eval_solution_list)]
                        # "reward"
                        eval_total += len(tmp_rewards)
                        eval_ans_true += sum(1 for r in tmp_rewards if r["answer_reward"] == 1)
                        eval_all_true += sum(1 for r in tmp_rewards if r["reward"] == 1)
                        print(f"Epoch {epoch}, Ans True: {eval_ans_true}, All True: {eval_all_true}, Total: {eval_total}")
                        
                eval_ans_accuracy = eval_ans_true / eval_total
                eval_all_accuracy = eval_all_true / eval_total
                wandb.log({
                    # "eval/loss": sum(eval_losses) / len(eval_losses),
                    "eval/ans_accuracy": eval_ans_accuracy,
                    "eval/all_accuracy": eval_all_accuracy,
                })
                print(f"Eval Loss: {sum(eval_losses) / len(eval_losses)}, Ans Accuracy: {eval_ans_accuracy}, All Accuracy: {eval_all_accuracy}")
    # end eval model
    del eval_model

if __name__ == "__main__":
    sft_main(n_epochs=10, bs=2, lr=1e-5, grad_accumulation=2, eval_step=200)
    # Note: Adjust the parameters as needed for your training setup.