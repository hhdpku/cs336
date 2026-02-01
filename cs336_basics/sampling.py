from typing import Optional

import torch

from model import TransformerLM
from tokenizer import BPETokenizer

@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: Optional[str] = None,
) -> torch.Tensor:
    model.eval()

    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)  # [1, T]

    if device is not None:
        prompt = prompt.to(device)
        model.to(device)

    generated = prompt  # [1, T]

    for _ in range(max_new_tokens):
        # 如果模型有 context_length，就做截断，防止超过上下文窗口
        if hasattr(model, "context_length"):
            context_len = model.context_length
            input_ids = generated[:, -context_len:]
        else:
            input_ids = generated

        # 前向：得到最后一个位置的 logits，形状 [1, vocab_size]
        logits = model(input_ids)[:, -1, :]

        # 温度缩放
        if temperature > 0:
            logits = logits / temperature

        # 温度为 0 时，直接贪心
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
        else:
            probs = torch.softmax(logits, dim=-1)  # [1, V]

            # top-p / nucleus sampling
            if top_p < 1.0:
                # 排序
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True, dim=-1
                )  # [1, V]
                cumsum = torch.cumsum(sorted_probs, dim=-1)

                # 找到累积概率超过 top_p 的位置，并 mask 掉
                mask = cumsum > top_p
                # 至少保留一个 token
                mask[..., 0] = False

                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                # 在截断后的分布上采样
                sampled_idx = torch.multinomial(sorted_probs, num_samples=1)  # [1, 1]
                next_token = sorted_indices.gather(-1, sampled_idx)  # [1, 1]
            else:
                # 普通 multinomial 采样
                next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

        # 把新 token 拼在序列后面
        generated = torch.cat([generated, next_token], dim=1)  # [1, T+1]

        # 如果全是 eos，就提前停止
        if (next_token == eos_token_id).all():
            break

    return generated

tokenizer = BPETokenizer.load("/root/llm-from-scratch-assignment1-basics/tokenizer/ls_vocab.tsv", "/root/llm-from-scratch-assignment1-basics/tokenizer/ls_merges.txt")
checkpoint = torch.load("checkpoint/model.pt", map_location="cuda")
model = TransformerLM(
    vocab_size=10000,
    context_length = 256,
    d_model=512,
    d_ff=1365,  # 改回这个数字
    num_layers=4,
    num_heads=16,
)
model.load_state_dict(checkpoint["model_state"])
model.to("cuda")

prompt_text = "Once upon a time"
prompt_ids = tokenizer.encode(prompt_text)
prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)

eos_id = tokenizer.vocab[b'<|endoftext|>']

out = generate(
    model,
    prompt=prompt_tensor,
    max_new_tokens=250,
    eos_token_id=eos_id,
    temperature=0.8,
    top_p=0.9,
    device="cuda",
)

generated_ids = out[0].tolist()
generated_text = tokenizer.decode(generated_ids)
print(generated_text)
# from typing import Optional

# import torch
# from tokenizers import Tokenizer

# from model import TransformerLM

# @torch.no_grad()
# def generate(
#     model: TransformerLM,
#     prompt: torch.Tensor,
#     max_new_tokens: int,
#     eos_token_id: int,
#     temperature: float = 1.0,
#     top_p: float = 1.0,
#     device: Optional[str] = None,
# ) -> torch.Tensor:
#     model.eval()

#     if prompt.dim() == 1:
#         prompt = prompt.unsqueeze(0)  # [1, T]

#     if device is not None:
#         prompt = prompt.to(device)
#         model.to(device)

#     generated = prompt  # [1, T]

#     for _ in range(max_new_tokens):
#         if hasattr(model, "context_length"):
#             context_len = model.context_length
#             input_ids = generated[:, -context_len:]
#         else:
#             input_ids = generated

#         logits = model(input_ids)[:, -1, :]  # [1, vocab_size]

#         if temperature > 0:
#             logits = logits / temperature

#         if temperature <= 0:
#             next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
#         else:
#             probs = torch.softmax(logits, dim=-1)

#             if top_p < 1.0:
#                 sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
#                 cumsum = torch.cumsum(sorted_probs, dim=-1)
#                 mask = cumsum > top_p
#                 mask[..., 0] = False  # 至少保留一个 token
#                 sorted_probs = sorted_probs.masked_fill(mask, 0.0)
#                 sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
#                 sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
#                 next_token = sorted_indices.gather(-1, sampled_idx)
#             else:
#                 next_token = torch.multinomial(probs, num_samples=1)

#         generated = torch.cat([generated, next_token], dim=1)

#         if (next_token == eos_token_id).all():
#             break

#     return generated

# # --------------------------
# # HuggingFace Tokenizer 加载
# # --------------------------
# tokenizer = Tokenizer.from_file(
#     "/root/llm-from-scratch-assignment1-basics/cs336_basics/owt_tokenizer.json"
# )

# # 模型加载
# checkpoint = torch.load("checkpoints/model.pt", map_location="cuda")
# model = TransformerLM(
#     vocab_size=tokenizer.get_vocab_size(),
#     context_length=256,
#     d_model=512,
#     d_ff=1365,
#     num_layers=4,
#     num_heads=16,
# )
# model.load_state_dict(checkpoint["model_state"])
# model.to("cuda")

# # --------------------------
# # prompt 编码
# # --------------------------
# prompt_text = "Once upon a time"
# prompt_ids = tokenizer.encode(prompt_text).ids  # HF tokenizer 返回对象里有 ids
# prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)

# # eos token id
# eos_id = tokenizer.token_to_id("<|endoftext|>")

# # --------------------------
# # 生成文本
# # --------------------------
# out = generate(
#     model,
#     prompt=prompt_tensor,
#     max_new_tokens=250,
#     eos_token_id=eos_id,
#     temperature=0.8,
#     top_p=0.9,
#     device="cuda",
# )

# generated_ids = out[0].tolist()
# generated_text = tokenizer.decode(generated_ids)
# print(generated_text)
