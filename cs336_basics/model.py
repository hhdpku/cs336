import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化权重矩阵 W
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
            
        self.reset_parameters()

    def reset_parameters(self):
        sigma = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=sigma, 
            a=-3.0 * sigma, 
            b=3.0 * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('...i, oi -> ...o', x, self.weight)

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # 初始化权重矩阵 W
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
            
        self.reset_parameters()

    def reset_parameters(self):
        sigma = math.sqrt(2.0 / (self.num_embeddings + self.embedding_dim))
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=sigma, 
            a=-3.0 * sigma, 
            b=3.0 * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(self,  d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.gain
        return result.to(in_dtype)
        # return x

def silu(x: torch.Tensor):
    return (x * torch.sigmoid(x))

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64

        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.W1(x)
        b = self.W3(x)
        gate = a * torch.sigmoid(a)
        return self.W2(gate * b)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        k = torch.arange(1, d_k // 2 + 1, device=device).float()
        exponent = -2 * (k - 1) / d_k
        inv_freq = theta ** exponent
        pos = torch.arange(max_seq_len, device=device).float()
        angles = torch.einsum('i, j -> ij', pos, inv_freq)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        x_rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        return x_rotated.flatten(-2)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    a = torch.einsum('...nk, ...mk -> ...nm', q, k)/(q.shape[-1] ** 0.5)
    if mask is not None:
        a = a.masked_fill(mask == False, float('-inf'))
    s = softmax(a, dim=-1)
    return torch.einsum('...nm, ...mv -> ...nv', s, v)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, rope: nn.Module, token_positions: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q = self.W_q(x).reshape(b, s, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).reshape(b, s, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).reshape(b, s, self.num_heads, self.d_k).transpose(1, 2)

        q = rope(q, token_positions.unsqueeze(1)) 
        k = rope(k, token_positions.unsqueeze(1))

        mask = torch.tril(torch.ones(s, s, device=x.device)).bool()

        out = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.W_o(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.rmsnorm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        
        self.rmsnorm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, rope: nn.Module, token_positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rmsnorm_1(x), rope, token_positions)
        x = x + self.ffn(self.rmsnorm_2(x))
        # attn_output = self.attn(x, rope, token_positions)
        # x = self.rmsnorm_1(x + attn_output)

        # ffn_output = self.ffn(x)
        # x = self.rmsnorm_2(x + ffn_output)
        return x
    
class TransformerLM(nn.Module):
    def __init__(
        self, vocab_size: int, context_length: int, num_layers: int, 
        d_model: int, num_heads: int, d_ff: int, device=None, dtype=None
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RoPE(
            theta=10000.0, d_k=d_model // num_heads, max_seq_len=context_length, device=device
        )
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s = token_ids.shape
        token_positions = torch.arange(s, device=token_ids.device).unsqueeze(0).repeat(b, 1)

        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x, self.rope, token_positions)

        x = self.final_norm(x)

        return self.output_head(x)
