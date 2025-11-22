import math
import time
import string
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Tokenizer (char-level, simple)
# ----------------------------

class CharTokenizer:
    """Very simple character-level tokenizer over printable ASCII."""
    def __init__(self):
        self.chars = sorted(list(set(string.printable)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos.get(i, '?') for i in ids)


# ----------------------------
# Tiny GPT-style model
# ----------------------------

@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, n_heads, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nH, T, dH)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, nH, T, T)

        # causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float('-inf'))

        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nH, T, dH)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.proj(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPTLM(nn.Module):
    """A tiny GPT-style language model with causal self-attention."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) integer token ids
        returns: logits (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, "Sequence length exceeds model max_seq_len"

        tok_emb = self.token_emb(idx)  # (B, T, C)
        pos_emb = self.pos_emb[:, :T, :]  # (1, T, C)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits


# ----------------------------
# Inference server
# ----------------------------

class InferenceServer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: CharTokenizer,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> Dict[str, Any]:
        """
        Returns dict with:
          - 'prompt'
          - 'output_text'
          - 'ttft_ms' (time-to-first-token)
          - 'avg_token_ms'
          - 'token_latencies_ms' (list)
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        if len(input_ids) == 0:
            input_ids = [0]

        # Truncate if too long
        max_seq_len = self.model.config.max_seq_len
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[-max_seq_len:]

        x = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, T)

        token_latencies_ms: List[float] = []

        # Choose timing mode
        use_cuda_timing = self.device.type == "cuda" and torch.cuda.is_available()

        if use_cuda_timing:
            torch.cuda.synchronize()
        else:
            # CPU: use wall-clock
            pass

        start_time_cpu = time.perf_counter()  # for total latency if you want it

        for step in range(max_new_tokens):
            if x.size(1) > max_seq_len:
                x = x[:, -max_seq_len:]  # crop to context window

            if use_cuda_timing:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()

                logits = self.model(x)  # (1, T, V)

                end_evt.record()
                torch.cuda.synchronize()
                step_ms = start_evt.elapsed_time(end_evt)
            else:
                t0 = time.perf_counter()
                logits = self.model(x)
                t1 = time.perf_counter()
                step_ms = (t1 - t0) * 1000.0

            token_latencies_ms.append(step_ms)

            # Take logits for last token
            logits_last = logits[:, -1, :] / max(temperature, 1e-5)  # (1, V)

            if top_k is not None and top_k > 0:
                # Top-k filtering
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                min_threshold = v[:, -1].unsqueeze(-1)
                logits_last = torch.where(
                    logits_last < min_threshold,
                    torch.full_like(logits_last, float('-inf')),
                    logits_last,
                )

            probs = F.softmax(logits_last, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # Append next token
            x = torch.cat([x, next_id], dim=1)

        # Compute metrics
        total_time_cpu_ms = (time.perf_counter() - start_time_cpu) * 1000.0
        ttft_ms = token_latencies_ms[0] if token_latencies_ms else None
        avg_token_ms = sum(token_latencies_ms) / len(token_latencies_ms) if token_latencies_ms else None

        # Decode everything *after* the original prompt as generated text
        generated_ids = x[0].tolist()[len(input_ids):]
        output_text = self.tokenizer.decode(generated_ids)

        return {
            "prompt": prompt,
            "output_text": output_text,
            "ttft_ms": ttft_ms,
            "avg_token_ms": avg_token_ms,
            "token_latencies_ms": token_latencies_ms,
            "total_time_ms": total_time_cpu_ms,
        }


# ----------------------------
# Main: simple REPL server
# ----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed for reproducibility
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    tokenizer = CharTokenizer()
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=256,
    )
    model = TinyGPTLM(config)
    server = InferenceServer(model, tokenizer, device)

    print("Tiny inference server ready. Type a prompt (or 'exit').\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if prompt.strip().lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = server.generate(
            prompt,
            max_new_tokens=32,
            temperature=1.0,
            top_k=50,
        )

        print("\n--- Generation ---")
        print(prompt + result["output_text"])
        print("\n--- Metrics ---")
        print(f"TTFT: {result['ttft_ms']:.3f} ms")
        print(f"Avg token latency: {result['avg_token_ms']:.3f} ms")
        print(f"Total time (CPU wall-clock): {result['total_time_ms']:.3f} ms")
        print("-----------------\n")


if __name__ == "__main__":
    main()