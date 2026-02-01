import torch
import torch.nn as nn
import math


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    m = torch.max(logits, dim=-1, keepdim=True).values
    log_sum_exp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(logits - m), dim=-1))
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = log_sum_exp - target_logits
    return loss.mean()
