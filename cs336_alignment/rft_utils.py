from typing import Callable, Literal
import torch
import torch.nn as nn

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.

    Args:
        reward_fn (Callable[[str, str], dict[str, float]]): Scores the rollout responses against
            the ground truths, producing a dict with keys "reward", "format_reward", and
            "answer_reward".
        rollout_responses (list[str]): Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths (list[str]): The ground truths for the examples. The length of this
            list is rollout_batch_size, because the ground truth for each example is repeated
            group_size times.
        group_size (int): Number of responses per question (group).
        advantage_eps (float): Small constant to avoid division by zero in normalization.
        normalize_by_std (bool): If True, divide by the per-group standard deviation; otherwise
            subtract only the group mean.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]: A tuple containing:
            - advantages (torch.Tensor): Shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
            - rewards (torch.Tensor): Shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
            - metadata (dict[str, float]): Additional information about the rewards, such as the group mean and std. Your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    assert len(rollout_responses) == len(repeated_ground_truths)
    meta_rewards = [reward_fn(resp, grth) for resp, grth in zip(rollout_responses, repeated_ground_truths)]
    rewards = torch.tensor([r["reward"] for r in meta_rewards], dtype=torch.float32)
    rewards = rewards.view(-1, group_size) # (n_prompts_per_rollout_batch, group_size)
    advantage = rewards - rewards.mean(dim=1, keepdim=True)  # (n_prompts_per_rollout_batch, group_size) # NOTE: NORMALIZE AMONG THE GROUP, NEVER AMONG THE BATCH
    if normalize_by_std:
        advantage /= rewards.std(dim=1, keepdim=True) + advantage_eps
    return advantage.view(-1), rewards.view(-1), {
        "mean": rewards.mean(1).tolist(),
        "std": rewards.std(1).tolist(),
        "max": rewards.max(1).values.tolist(),
        "min": rewards.min(1).values.tolist(),
        "meta_rewards": meta_rewards,
    }
    
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either the raw reward or an already-normalized advantage.

    Args:
        raw_rewards_or_advantages (torch.Tensor): Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs (torch.Tensor): Shape (batch_size, sequence_length), logprobs for each token.

    Returns:
        torch.Tensor: Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
        be aggregated across the batch and sequence dimensions in the training loop).
    """
    return -(raw_rewards_or_advantages * policy_log_probs) # per-token per-batch loss

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages (torch.Tensor): Shape (batch_size, 1), per-example advantages A.
        policy_log_probs (torch.Tensor): Shape (batch_size, sequence_length), per-token log
            probs from the policy being trained.
        old_log_probs (torch.Tensor): Shape (batch_size, sequence_length), per-token log probs
            from the old policy.
        cliprange (float): Clip parameter ϵ (epsilon, e.g. 0.2).
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss (torch.Tensor): Shape (batch_size, sequence_length), the per-token clipped
                loss.
            metadata (dict[str, torch.Tensor]): dict containing whatever you want to log. We suggest logging whether each
                token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
                the min was lower than the LHS.
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)  # (batch_size, sequence_length)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    is_clipped = (clipped_ratio != ratio).detach().cpu()
    clipped_loss = -torch.min(ratio * advantages.view(-1, 1), clipped_ratio * advantages.view(-1, 1))
    return clipped_loss, {"is_clipped": is_clipped, "clip_fraction": is_clipped.float().mean().item()}

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    Args:
        policy_log_probs (torch.Tensor): Shape (batch_size, sequence_length), per-token log-probabilities from the
            policy being trained.
        loss_type (Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]): The type of loss to compute.
        raw_rewards (torch.Tensor | None): Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages (torch.Tensor | None): Required for "reinforce_with_baseline" and "grpo_clip"; shape
            (batch_size, 1).
        old_log_probs (torch.Tensor | None): Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange (float | None): Required for "grpo_clip"; scalar ϵ used for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss (torch.Tensor): Shape (batch_size, sequence_length), per-token loss.
            metadata (dict[str, torch.Tensor]): dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards.to(policy_log_probs.device), policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages.to(policy_log_probs.device), policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages.to(policy_log_probs.device),
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs.to(policy_log_probs.device),
            cliprange=cliprange
        )
        return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1

    Args:
        tensor (torch.Tensor): The data to be averaged.
        mask (torch.Tensor): Same shape as tensor; positions with 1 are included in the mean.
        dim (int | None): Dimension over which to average. If None, compute the mean over all masked elements.

    Returns:
        torch.Tensor: The masked mean of the input tensor.
    """
    # XXX: return (tensor * mask).mean(dim=dim) # it's not 1/N. it's 1/N_mask_is_1
    n_mask_is_1 = mask.sum(dim)
    return (tensor * mask).sum(dim) / n_mask_is_1

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    
    Args:
        policy_log_probs (torch.Tensor): Shape (batch_size, sequence_length), per-token log-probabilities from the
            policy being trained.
        response_mask (torch.Tensor): Shape (batch_size, sequence_length), 1 for response tokens, 0 for
            prompt/padding.
        gradient_accumulation_steps (int): Number of microbatches per optimizer step.
        loss_type (Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]): The type of loss to compute.
        raw_rewards (torch.Tensor | None): Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages (torch.Tensor | None): Required for "reinforce_with_baseline" and "grpo_clip"; shape
            (batch_size, 1).
        old_log_probs (torch.Tensor | None): Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange (float | None): Required for "grpo_clip"; scalar ϵ used for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss (torch.Tensor): Shape (batch_size, sequence_length), per-token loss.
            metadata (dict[str, torch.Tensor]): dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    old_log_probs = old_log_probs.detach() # stop grad
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )
    
    loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    loss.backward()
    metadata["policy_log_probs_grad"] = policy_log_probs.retain_grad()
    return loss, metadata