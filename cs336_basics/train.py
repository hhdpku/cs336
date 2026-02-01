import os
import time
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

from model import TransformerLM
from loss import cross_entropy
from optimizer import AdamW, learning_rate_schedule, gradient_clipping
from dataloader import data_loader

def train(
    train_bin_path: str,
    val_bin_path: str,
    vocab_size: int,
    context_length: int = 128,
    num_layers: int = 4,
    d_model: int = 256,
    num_heads: int = 8,
    d_ff: int | None = None,
    batch_size: int = 32,
    max_steps: int = 10_000,
    lr_max: float = 3e-4,
    lr_min: float = 3e-5,
    warmup_steps: int = 1_000,
    lr_decay_steps: int = 10_000,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    eval_interval: int = 500,
    eval_iters: int = 50,
    checkpoint_path: str | None = None,
    checkpoint_interval: int = 1_000,
    device: str | None = None,
    data_dtype=np.uint16,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 内存映射数据
    train_data = np.memmap(train_bin_path, dtype=data_dtype, mode="r")
    val_data = np.memmap(val_bin_path, dtype=data_dtype, mode="r")

    # 构建模型
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff if d_ff is not None else int(8/3*d_model),
        device=device,
        dtype=torch.float32,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)
    start_step = 0

    # # 恢复 checkpoint
    # if checkpoint_path and os.path.exists(checkpoint_path):
    #     ckpt = torch.load(checkpoint_path, map_location=device)
    #     model.load_state_dict(ckpt["model_state"])
    #     optimizer.load_state_dict(ckpt["optimizer_state"])
    #     start_step = ckpt.get("step", 0)
    #     print(f"Loaded checkpoint from {checkpoint_path} at step {start_step}")

    # 用于绘图的 loss 历史
    step_history = []
    train_loss_history = []
    val_loss_history = []
    start_time = time.time()

    @torch.no_grad()
    def estimate_loss(data_array: np.memmap) -> float:
        model.eval()
        losses = []
        for _ in range(eval_iters):
            X, Y = data_loader(data_array, batch_size=batch_size, context_length=context_length, device=device)
            logits = model(X)
            loss = cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    for step in range(start_step, max_steps):
        # 学习率调度
        lr = learning_rate_schedule(step, alpha_max=lr_max, alpha_min=lr_min, Tw=warmup_steps, Tc=lr_decay_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        X, Y = data_loader(train_data, batch_size=batch_size, context_length=context_length, device=device)
        logits = model(X)
        loss = cross_entropy(logits.view(-1, vocab_size), Y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip:
            gradient_clipping(model.parameters(), grad_clip)

        optimizer.step()

        # 每 100 step 打印一次训练 loss
        if (step + 1) % 100 == 0 or step == 0:
            print(f"step {step+1}/{max_steps} | train_loss {loss.item():.4f} | lr {lr:.2e}")

        # 每 eval_interval 评估一次
        if (step + 1) % eval_interval == 0:
            train_loss = estimate_loss(train_data)
            val_loss = estimate_loss(val_data)
            elapsed = time.time() - start_time
            print(f"[eval] step {step+1} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

            # 保存 loss 历史
            step_history.append(step + 1)
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

        # 周期性保存 checkpoint
        if checkpoint_path and (step + 1) % checkpoint_interval == 0:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step+1,
                "config": {
                    "vocab_size": vocab_size,
                    "context_length": context_length,
                    "num_layers": num_layers,
                    "d_model": d_model,
                    "num_heads": num_heads,
                    "d_ff": d_ff,
                }
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path} at step {step+1}")

    print("Training finished.")

    # 训练结束画 loss 曲线
    if step_history:
        plt.figure(figsize=(8,5))
        plt.plot(step_history, train_loss_history, label="Train Loss")
        plt.plot(step_history, val_loss_history, label="Validation Loss")
        plt.xlabel("Gradient Steps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plot_name = f"loss_curve.png"
        plt.savefig(plot_name)
        print(f"Plot saved as {plot_name}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--lr_max", type=float, default=5e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--lr_decay_steps", type=int, default=10_000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=1_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data_dtype", type=str, default="uint16")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dtype_map = {
        "uint16": np.uint16,
        "int32": np.int32,
        "int64": np.int64,
    }
    data_dtype = dtype_map.get(args.data_dtype, np.uint16)

    train(
        train_bin_path=args.train_data,
        val_bin_path=args.val_data,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        warmup_steps=args.warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        checkpoint_path=args.checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        data_dtype=data_dtype,
    )