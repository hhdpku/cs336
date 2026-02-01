from collections.abc import Callable, Iterable
from typing import Optional
import torch
import torch.nn as nn
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data-= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0: 
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: 
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: 
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1
                denom = torch.sqrt(v).add(eps)
                p.data.addcdiv_(m, denom, value=-alpha_t)

                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)
        return loss

def learning_rate_schedule(t, alpha_max, alpha_min, Tw, Tc):
    if t < Tw:
        return (t / Tw) * alpha_max
    elif t <= Tc:
        cosine_term = math.cos((t - Tw) / (Tc - Tw) * math.pi)
        return alpha_min + 0.5 * (1 + cosine_term) * (alpha_max - alpha_min)
    else:
        return alpha_min

def gradient_clipping(parameters, max_norm, eps=1e-6):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.data.pow(2).sum().item()
    total_norm = math.sqrt(total)
    if total_norm <= max_norm:
        return
    scale = max_norm / (total_norm + eps)
    for p in parameters:
        if p.grad is not None:
            p.grad.data.mul_(scale)