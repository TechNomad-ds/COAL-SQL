import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F

def compute_sft_loss(log_prob, eos_mask):
    sft_losses = -1 * log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }   

# lower entropy tokens with higher weight
def compute_sft_loss_v1(log_prob, eos_mask, entropy):
    sft_losses = -1 * log_prob
    weight = 0.5 * torch.exp(-entropy.detach())
    sft_losses = weight * sft_losses
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }  

# loss reshaping following LUFFY (https://arxiv.org/pdf/2504.14945)
def compute_sft_loss_v2(log_prob, eos_mask):
    prob = torch.exp(log_prob)
    shaped_prob = prob/(prob + 0.1)
    
    sft_losses = -1 * shaped_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }   

# higher entropy tokens with higher weight
def compute_sft_loss_v3(log_prob, eos_mask, entropy):
    sft_losses = -1 * log_prob
    weight = 0.5 * torch.exp(entropy.detach())
    sft_losses = weight * sft_losses
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }  

# only update tokens with lower entropy
def compute_sft_loss_v4(log_prob, eos_mask, entropy, ratio=0.5):
    sft_losses = -1 * log_prob

    masked_entropy = entropy.clone()
    masked_entropy[~eos_mask] = float('inf')

    flat_entropy = masked_entropy.view(-1)
    num_valid = eos_mask.sum().item()
    k = max(1, int(num_valid * ratio))

    topk_entropy, _ = torch.topk(flat_entropy, k, largest=False)
    threshold = topk_entropy[-1]

    # Build mask: only positions with entropy <= threshold are True
    selected_mask = (masked_entropy <= threshold) & eos_mask

    # Compute loss only over positions where selected_mask is True
    sft_loss = verl_F.masked_mean(sft_losses, selected_mask)
    return {
        "sft_loss": sft_loss,
    }

# only update tokens with high entropy
def compute_sft_loss_v5(log_prob, eos_mask, entropy, ratio=0.2):
    sft_losses = -1 * log_prob

    masked_entropy = entropy.clone()
    masked_entropy[~eos_mask] = float('-inf')

    flat_entropy = masked_entropy.view(-1)
    num_valid = eos_mask.sum().item()
    k = max(1, int(num_valid * ratio))  # select at least 1

    topk_entropy, _ = torch.topk(flat_entropy, k, largest=True)
    threshold = topk_entropy[-1]

    selected_mask = (masked_entropy >= threshold) & eos_mask

    sft_loss = verl_F.masked_mean(sft_losses, selected_mask)
    return {
        "sft_loss": sft_loss,
    }


def compute_sft_loss_v6(log_prob, eos_mask, entropy, low_ratio=0.25, high_ratio=0.75):
    """
    Compute SFT loss only over EOS tokens whose entropy falls within
    [low_ratio, high_ratio]. Uses two ascending torch.topk calls to obtain the
    25% and 75% thresholds, avoiding torch.sort.
    """
    sft_losses = -log_prob                                # [B, L]

    # 1. Gather entropy of all valid tokens
    eos_mask = eos_mask.bool()
    valid_entropy = entropy[eos_mask]                    # [N]

    N = valid_entropy.numel()
    if N == 0:
        sft_loss = torch.tensor(0.0, device=log_prob.device, requires_grad=True)
        return {"sft_loss": sft_loss}

    # 2. Compute the 25% and 75% thresholds (ascending topk)
    k_low  = max(1, int(N * low_ratio))
    k_high = max(1, int(N * high_ratio))

    # Ascending topk, take the k_low-th (largest=False)
    _, idx_low = torch.topk(valid_entropy, k_low, largest=False)
    low_th = valid_entropy[idx_low[-1]]

    # Ascending topk, take the k_high-th
    _, idx_high = torch.topk(valid_entropy, k_high, largest=False)
    high_th = valid_entropy[idx_high[-1]]

    # 3. Build mask
    selected_mask = (entropy >= low_th) & (entropy <= high_th) & eos_mask

    # 4. Compute mean loss
    sft_loss = verl_F.masked_mean(sft_losses, selected_mask)
    return {"sft_loss": sft_loss}

# only update tokens with lower entropy but per sentence
def compute_sft_loss_v4_per_sentence(log_prob, eos_mask, entropy, ratio=0.2):
    sft_losses = -1 * log_prob  # [B, T]

    B, T = entropy.shape
    selected_mask = torch.zeros_like(entropy, dtype=torch.bool)

    for i in range(B):
        eos_i = eos_mask[i].bool()  # [T]
        ent_i = entropy[i][eos_i]  # keep only entropy of valid tokens
        num_valid = ent_i.numel()
        k = max(1, int(num_valid * ratio))
        topk_ent, _ = torch.topk(ent_i, k, largest=False)
        threshold = topk_ent[-1]

        # Build the mask for this sentence
        mask_i = (entropy[i] <= threshold) & eos_i
        selected_mask[i] = mask_i

    # Compute loss only over positions where selected_mask is True
    sft_loss = verl_F.masked_mean(sft_losses, selected_mask)
    return {
        "sft_loss": sft_loss,
    }

# only update tokens with high entropy but per sentence
def compute_sft_loss_v5_per_sentence(log_prob, eos_mask, entropy, ratio=0.2):
    sft_losses = -1 * log_prob  # [B, T]

    B, T = entropy.shape
    selected_mask = torch.zeros_like(entropy, dtype=torch.bool)

    for i in range(B):
        # Get valid tokens of the i-th sentence
        eos_i = eos_mask[i].bool()  # [T]
        # Keep only entropy of valid tokens
        ent_i = entropy[i][eos_i]

        num_valid = ent_i.numel()
        # Select at least one token
        k = max(1, int(num_valid * ratio))

        # Find the top-k highest-entropy tokens in the current sentence
        topk_ent, _ = torch.topk(ent_i, k, largest=True)
        # Use the smallest entropy among the top-k as the threshold
        threshold = topk_ent[-1]

        # Build the selection mask for this sentence:
        # select tokens with entropy >= threshold that are also valid
        mask_i = (entropy[i] >= threshold) & eos_i
        selected_mask[i] = mask_i

    # Compute loss only over positions where selected_mask is True
    sft_loss = verl_F.masked_mean(sft_losses, selected_mask)
    return {
        "sft_loss": sft_loss,
    }

def compute_sft_loss_v7(log_prob, eos_mask):
    prob = torch.exp(log_prob).detach()
    sft_losses = -1 * prob * log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }   

def compute_sft_loss_v8(log_prob, eos_mask):
    prob = torch.exp(log_prob).detach()
    sft_losses = -1 * prob * (1-prob) * log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return {
        "sft_loss": sft_loss,
    }   
