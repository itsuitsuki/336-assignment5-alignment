from vllm_utils import init_vllm, load_policy_into_vllm_instance
from vllm import SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
import torch
from cs336_alignment.rft_utils import d
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import wandb
from tqdm import tqdm

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def dpo_hh_rlhf(args):
    wandb.init(project="cs336_alignment", 
            name=f"dpo_hh_rlhf",
            config=vars(args))
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    device_policy = "cuda:0"
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map={"": "cuda:0"}, # if specify on 1 then the device_map should be "cuda:1"
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device_policy)
    device_ref = "cuda:1"
    # same
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map={"": "cuda:1"}, # if specify on 1 then the device_map should be "cuda:1"
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device_ref)
    # freeze params for ref
    for param in ref_model.parameters():
        param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        max_length=args.max_len
    )
    subsets = ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"]
    hh_dataset = concatenate_datasets(
        [load_dataset("Anthropic/hh-rlhf", data_dir=subset)["train"] for subset in subsets]
    )
    # shuffle + valid 200 examples
    hh_dataset = hh_dataset.shuffle(seed=args.seed)
    train_dataset = hh_dataset.select([i for i in list(range(200))])
    