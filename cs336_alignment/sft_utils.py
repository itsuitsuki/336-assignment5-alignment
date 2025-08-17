import json
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import torch
import torch.nn.functional as F
from datasets import load_dataset
import re

def naive_load_model_and_tokenizer():
    # qwen 2.5 math 1.5b
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        trust_remote_code=True,
    )
    # print(type(tokenizer))
    return model, tokenizer

def demo_forward_pass(batch, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = batch["input_ids"].to(device) # (B, L)
    labels = batch["labels"].to(device) # (B, L)
    logits = model(input_ids).logits # (B, L, V)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    
    
    print("loss: ", loss.item())

# def demo():
#     ds = load_math()["train"]
#     prompt = open("cs336_alignment/prompts/r1_zero.prompt", "r").read()
#     model, tokenizer = naive_load_model_and_tokenizer()
#     problem = next(iter(ds))["problem"]
#     prefill = prompt.format(question=problem)
#     dummy_response = "Yes. It's 42. </think>\n<answer>42</answer>"  # Dummy answer for demonstration
#     prefill_ids = tokenizer(prefill, return_tensors="pt").input_ids
#     response_ids = tokenizer(dummy_response, return_tensors="pt").input_ids

#     # input_ids = tokenizer(model_input, return_tensors="pt").input_ids # pt means pytorch
#     input_ids = torch.cat([prefill_ids, response_ids], dim=1)  # Concatenate prefill and response

#     # assert tokenizer(prefill + dummy_response, return_tensors="pt").input_ids.shape == input_ids.shape, "Input IDs do not match concatenated input."
#     # print("input ids:", input_ids)
#     # print("concat ids:", tokenizer(prefill + dummy_response, return_tensors="pt").input_ids)
#     labels = input_ids.clone() # labels are the same as input_ids for language modeling
#     # [:-1] and [1:]
#     input_ids = input_ids[:, :-1]  # remove last token
#     labels = labels[:, 1:]  # remove first token

#     # forward pass
#     demo_forward_pass({"input_ids": input_ids, "labels": labels}, model)

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer):
    """
    prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs (lists[str]): _description_
        output_strs (lists[str]): _description_
        tokenizer (PreTrainedTokenizer): _description_
        
    Returns: dict[str, torch.Tensor] where the keys are corresponding to:
        input_ids (torch.Tensor): shape (batch_size, max(prompt_and_output_lens)-1): the tokenized input ids w/ the final token sliced off
        labels (torch.Tensor): shape (batch_size, max(prompt_and_output_lens)-1): the tokenized labels w/ the first token sliced off
        response_mask (torch.Tensor): shape (batch_size, max(prompt_and_output_lens)-1): 1 for response tokens, 0 for prompt or padding tokens
    """
    tokenizer_out = tokenizer(prompt_strs, output_strs, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
    # hand add eos token to the end of each output
    batch_size = tokenizer_out.input_ids.shape[0]
    
    # DONT ADD THE EXTRA EOS!
    # eos_column = torch.full((batch_size, 1), tokenizer.eos_token_id, device=tokenizer_out.input_ids.device)
    # attn_column = torch.ones((batch_size, 1), device=tokenizer_out.attention_mask.device, dtype=tokenizer_out.attention_mask.dtype)
    # tokenizer_out.input_ids = torch.cat([tokenizer_out.input_ids, eos_column], dim=1)
    # tokenizer_out.attention_mask = torch.cat([tokenizer_out.attention_mask, attn_column], dim=1)

    ori_input_ids = tokenizer_out.input_ids  # shape (batch_size, max(prompt_and_output_lens))

    input_ids = ori_input_ids[:, :-1].clone()  # remove last token
    labels = ori_input_ids[:, 1:].clone()  # remove first token
    
    prompt_lens = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompt_strs]
    
    response_mask = tokenizer_out.attention_mask
    for i, prompt_len in enumerate(prompt_lens):
        # padding also has 0 in response_mask
        response_mask[i, :prompt_len] = 0
        # set pad to 0
    response_mask = response_mask[:, 1:]
        
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of the logits along the last dimension.

    Args:
        logits (torch.Tensor): The logits tensor of shape (batch_size, sequence_length, vocab_size).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, sequence_length) containing the entropy for each token.
    """
    with torch.no_grad():
        probs = F.softmax(logits.detach().cpu(), dim=-1) # on cpu
        log_probs = torch.log(probs + 1e-10)  # Add a small value to avoid log(0)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy # shape: (batch_size, sequence_length)

def get_response_log_probs(model, input_ids: torch.Tensor, labels: torch.Tensor, return_token_entropy: bool = False) -> dict[str, torch.Tensor]:
    """
    Get the log probabilities of the response tokens in the model's output.

    Args:
        model (PreTrainedModel | LLM): The pre-trained model.
        input_ids (torch.Tensor): The input IDs of shape (batch_size, sequence_length).
        labels (torch.Tensor): The labels of shape (batch_size, sequence_length).
        return_token_entropy (bool): Whether to return the token entropy.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing:
            - "log_probs": Log probabilities of the response tokens.
            - "token_entropy": Entropy of the response tokens if requested.
    """
    logits = model.forward(input_ids, labels=labels, output_hidden_states=False).logits
    result = {}
    if return_token_entropy:
        entropy = compute_entropy(logits)
        # assert on cpu
        # assert result["token_entropy"].device.type == "cpu"  
    # Get the log probabilities for the response tokens
    log_probs = F.log_softmax(logits, dim=-1)
    del logits
    
    # Extract the log probabilities for the response tokens
    response_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # shape (batch_size, sequence_length)
    del log_probs
    result["log_probs"] = response_log_probs
    if return_token_entropy:
        result["token_entropy"] = entropy
    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    # sum and div by a const
    # actually not normalizing
    
    # sum only respecting a mask w/ m=1
    if dim is None:
        dim = tuple(range(tensor.dim()))
    masked_sum = torch.sum(tensor * mask, dim=dim, keepdim=False)
    
    return masked_sum / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # policy_log_probs: (B, L), per-token log probabilities from SFT policy being trained.
    # response_mask: (B, L), 1 for response tokens, 0 for prompt or padding tokens.
    # gradient_accumulation_steps: int, number of steps to accumulate gradients.
    # normalize_constant: float, constant to normalize the loss.
    
    # Compute the loss = masked sum of log probabilities
    loss = -masked_normalize(
        tensor=policy_log_probs,  # negative because we want to maximize the log probabilities
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=1,  # sum over all.
    ) # shape (B, )
    # if dim=1 then # (B, ) where every element is the sum of log probabilities for the response tokens in one example.

    # n_effective_tokens = response_mask.sum()
    # loss /= n_effective_tokens
    loss = loss.mean()
    loss = loss / gradient_accumulation_steps  # average over the gradient accumulation steps
    
    loss.backward()  # Backpropagate the loss

    # return loss and metadata
    return loss, {"policy_log_probs_grad": policy_log_probs.retain_grad()}

def sft_microbatch_loss_wo_grad(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """
    Compute the SFT microbatch loss without gradient computation.
    """
    loss = masked_normalize(
        tensor= -policy_log_probs,  # negative because we want to maximize the log probabilities
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=1,  # sum over all.
    )
    loss = loss.mean()
    return loss

def log_generations(
    input_prompt: str | list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    groundtruth_answer: str | list[str],
    reward_fn: callable = None,
    reward: dict[str, float] = None,
    log_file: str = "generations.log",
) -> None:
    """
    Log the generations of the model for a given input prompt.
    Log these: 
    1. the input prompt(s), 
    2. the model's generated answer,
    3. the groundtruth answer(s),
    4. the reward (if provided). override reward_fn if reward is provided.
    5. average token entropy of the response.
    6. average response length, average response length for correct responses, and average response length for incorrect responses.
    """
    pass


def filter_dataset_by_answer_correctness(
    dataset,
    reward_fn: callable,
):
    """
    Filter the dataset by the correctness of the answers using the provided reward function.
    Args:
        dataset: The dataset to filter.
        reward_fn: A function that takes a model output and returns a reward score.
    Returns:
        A filtered dataset containing only the examples with correct answers.
    """
    filtered_dataset = []
    for example in dataset:
        answer = example["solution"]
        reward = reward_fn(answer)