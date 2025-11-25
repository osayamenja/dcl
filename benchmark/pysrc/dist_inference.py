import argparse
import math
import os
import time
import statistics
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ----------------------------
# Config and model components
# ----------------------------

@dataclass
class DistGPTConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 2048
    max_seq_len: int = 512


class TensorParallelSelfAttention(nn.Module):
    """
    Tensor-parallel self-attention:
      - Heads are split across ranks: each rank owns n_heads_local heads.
      - QKV projection and attention are local to each rank.
      - Output projection maps local heads back to d_model and then
        an all_reduce SUM combines partial outputs across ranks.
    """

    def __init__(self, config: DistGPTConfig, world_size: int):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % world_size == 0

        self.world_size = world_size
        self.n_heads_total = config.n_heads
        self.n_heads_local = config.n_heads // world_size
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.d_model_local = self.n_heads_local * self.head_dim

        # QKV projection produces 3 * d_model_local features per token
        self.qkv = nn.Linear(config.d_model, 3 * self.d_model_local, bias=False)
        # Output projection maps local heads back to full d_model
        self.proj = nn.Linear(self.d_model_local, config.d_model, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x: (B, T, d_model)
        past_kv: tuple (k, v) where each is (B, n_heads_local, T_past, head_dim)

        Note on masking:
          - Prefill (past_kv is None): we use is_causal=True.
          - Decode with KV cache (past_kv not None): we concatenate past+current
            keys/values and DO NOT use is_causal, since all keys are <= current
            position by construction (no future tokens are present).
        """
        B, T, C = x.shape
        device = x.device

        # Project to QKV for local heads
        qkv = self.qkv(x)  # (B, T, 3 * d_model_local)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, d_model_local)

        # Reshape to (B, n_heads_local, T, head_dim)
        q = q.view(B, T, self.n_heads_local, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads_local, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads_local, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            # Decode with KV cache: concatenate past and current keys/values.
            # Shapes:
            #   past_k, past_v: (B, n_heads_local, T_past, head_dim)
            #   k, v (current): (B, n_heads_local, T_cur,  head_dim)
            # After concat:
            #   k, v: (B, n_heads_local, T_total, head_dim)
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

            # No is_causal here: all keys are <= current logical position.
            # There are no "future" keys in k/v when using KV cache this way.
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                is_causal=False,
            )
        else:
            # Prefill: full sequence, standard causal masking.
            # q, k, v: (B, n_heads_local, T, head_dim)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                is_causal=True,
            )

        # attn_out: (B, n_heads_local, T_cur, head_dim)
        B_out, nH, T_cur, d = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(B_out, T_cur, self.d_model_local)

        # Local output projection -> partial output in d_model
        out_partial = self.proj(attn_out)  # (B, T_cur, d_model)

        # All-reduce partial results across ranks
        if self.world_size > 1:
            dist.all_reduce(out_partial, op=dist.ReduceOp.SUM)

        new_kv = None
        if use_cache:
            # Cache full K/V for this layer on this rank
            new_kv = (k, v)

        return out_partial, new_kv


class TensorParallelMLP(nn.Module):
    """
    Tensor-parallel MLP:
      - First linear expands d_model -> d_ff_local on each rank.
      - GELU activation is local.
      - Second linear projects d_ff_local -> d_model.
      - All-reduce SUM accumulates contributions across ranks.
    """
    def __init__(self, config: DistGPTConfig, world_size: int):
        super().__init__()
        assert config.d_ff % world_size == 0
        self.world_size = world_size
        self.d_model = config.d_model
        self.d_ff_total = config.d_ff
        self.d_ff_local = config.d_ff // world_size

        self.fc1 = nn.Linear(self.d_model, self.d_ff_local, bias=False)
        self.fc2 = nn.Linear(self.d_ff_local, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        h_local = self.fc1(x)  # (B, T, d_ff_local)
        h_local = F.gelu(h_local)
        out_partial = self.fc2(h_local)  # (B, T, d_model)

        if self.world_size > 1:
            dist.all_reduce(out_partial, op=dist.ReduceOp.SUM)

        return out_partial


class TPBlock(nn.Module):
    """Single transformer block with TP attention + MLP."""

    def __init__(self, config: DistGPTConfig, world_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = TensorParallelSelfAttention(config, world_size)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = TensorParallelMLP(config, world_size)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # x: (B, T, d_model)
        # Attention
        x_norm = self.ln1(x)
        attn_out, new_kv = self.attn(x_norm, past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out

        # MLP
        x_norm = self.ln2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out

        return x, new_kv


class DistributedGPT(nn.Module):
    """
    GPT-style LM with tensor-parallel attention/MLP across ranks.
    We use random weights; semantics don't matter, only compute + comm.
    """

    def __init__(self, config: DistGPTConfig, world_size: int):
        super().__init__()
        self.config = config
        self.world_size = world_size

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.d_model)
        )

        self.blocks = nn.ModuleList(
            [TPBlock(config, world_size) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        input_ids: (B, T)
        past_kv: list of length n_layers or None. Each element None or (k, v)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len

        # Embedding + positional encodings
        tok = self.token_emb(input_ids)  # (B, T, d_model)
        pos = self.pos_emb[:, :T, :]     # (1, T, d_model)
        x = tok + pos

        new_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        if use_cache:
            new_past = []

        for i, block in enumerate(self.blocks):
            layer_past = None
            if past_kv is not None:
                layer_past = past_kv[i]

            x, new_kv = block(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache and new_past is not None:
                new_past.append(new_kv)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        return logits, new_past


# ----------------------------
# Distributed inference harness
# ----------------------------

def run_once_dist(
    model: DistributedGPT,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    rank: int,
    world_size: int,
) -> Optional[Dict[str, Any]]:
    """
    Run one prefill + decode pass in a distributed way.
    Returns metrics dict on rank 0, None on other ranks.
    """
    device = next(model.parameters()).device
    use_cuda_timing = device.type == "cuda" and torch.cuda.is_available()

    if use_cuda_timing:
        torch.cuda.synchronize(device)

    # We'll measure E2E wall-clock on rank 0
    if rank == 0:
        total_start = time.perf_counter()

    # -------------------
    # Prefill: full prompt
    # -------------------
    prefill_ms = 0.0

    if use_cuda_timing and rank == 0:
        pre_s = torch.cuda.Event(enable_timing=True)
        pre_e = torch.cuda.Event(enable_timing=True)
        pre_s.record()

    logits, past_kv = model(prompt_ids, past_kv=None, use_cache=True)

    if use_cuda_timing and rank == 0:
        pre_e.record()
        torch.cuda.synchronize(device)
        prefill_ms = pre_s.elapsed_time(pre_e)
    elif rank == 0:
        # CPU fallback (unlikely in your setup)
        t0 = time.perf_counter()
        # (Already executed above, so this is just a placeholder for structure)
        t1 = time.perf_counter()
        prefill_ms = (t1 - t0) * 1000.0

    # -------------------
    # Decode loop
    # -------------------
    token_latencies_ms: List[float] = []

    # Initial token from prefill logits (last position)
    if rank == 0:
        logits_last = logits[:, -1, :]  # (1, vocab)
        # Greedy for simplicity; you can replace with sampling if you like
        next_token = torch.argmax(logits_last, dim=-1, keepdim=True)  # (1, 1)
    else:
        next_token = torch.empty(
            (1, 1), dtype=torch.long, device=device
        )

    # Broadcast next_token to all ranks
    dist.broadcast(next_token, src=0)

    # Decode step-by-step with KV cache
    for step in range(max_new_tokens):
        # Measure per-step latency on rank 0
        if use_cuda_timing and rank == 0:
            step_s = torch.cuda.Event(enable_timing=True)
            step_e = torch.cuda.Event(enable_timing=True)
            step_s.record()

        logits_step, past_kv = model(next_token, past_kv=past_kv, use_cache=True)

        if use_cuda_timing and rank == 0:
            step_e.record()
            torch.cuda.synchronize(device)
            step_ms = step_s.elapsed_time(step_e)
            token_latencies_ms.append(step_ms)

        # Choose next token based on latest logits
        if rank == 0:
            logits_last = logits_step[:, -1, :]  # (1, vocab)
            next_token = torch.argmax(logits_last, dim=-1, keepdim=True)
        # Other ranks receive the sampled token id
        dist.broadcast(next_token, src=0)

    if rank == 0:
        # E2E wall-clock
        total_ms = (time.perf_counter() - total_start) * 1000.0

        # Metrics
        ttft_ms = prefill_ms + (token_latencies_ms[0] if token_latencies_ms else 0.0)
        avg_token_ms = (
            sum(token_latencies_ms) / len(token_latencies_ms)
            if token_latencies_ms
            else 0.0
        )

        total_decode_ms = sum(token_latencies_ms)
        total_decode_s = total_decode_ms / 1000.0 if total_decode_ms > 0.0 else float("inf")
        total_e2e_s = total_ms / 1000.0 if total_ms > 0.0 else float("inf")

        decode_tokens_per_s = (
            max_new_tokens / total_decode_s if total_decode_s > 0.0 else 0.0
        )
        e2e_tokens_per_s = (
            max_new_tokens / total_e2e_s if total_e2e_s > 0.0 else 0.0
        )

        return {
            "prefill_ms": prefill_ms,
            "ttft_ms": ttft_ms,
            "avg_token_ms": avg_token_ms,
            "token_latencies_ms": token_latencies_ms,
            "total_ms": total_ms,
            "decode_tokens_per_s": decode_tokens_per_s,
            "e2e_tokens_per_s": e2e_tokens_per_s,
        }

    # Non-zero ranks don't report metrics
    return None


def run_benchmark_dist(
    config: DistGPTConfig,
    iterations: int,
    warmup_iters: int,
    min_new_tokens: int,
    max_new_tokens: int,
    step_new_tokens: int,
    prompt_len: int,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0 + rank)
    torch.cuda.manual_seed_all(0 + rank)

    # Build model
    model = DistributedGPT(config, world_size=world_size).to(device)
    model.eval()

    # Create synthetic prompt tokens, same on all ranks
    prompt_ids = torch.empty(
        (1, prompt_len),
        dtype=torch.long,
        device=device,
    )
    if rank == 0:
        prompt_ids.random_(0, config.vocab_size)
    dist.broadcast(prompt_ids, src=0)

    # CSV header printed by rank 0
    if rank == 0:
        print(
            "world_size,"
            "d_model,"
            "n_heads,"
            "n_layers,"
            "d_ff,"
            "vocab_size,"
            "prompt_len,"
            "max_new_tokens,"
            "iterations,"
            "warmup_iters,"
            "prefill_ms_mean,"
            "prefill_ms_std,"
            "ttft_ms_mean,"
            "ttft_ms_std,"
            "tpot_ms_mean,"
            "tpot_ms_std,"
            "total_ms_mean,"
            "total_ms_std,"
            "decode_tokens_per_s_mean,"
            "decode_tokens_per_s_std,"
            "e2e_tokens_per_s_mean,"
            "e2e_tokens_per_s_std"
        )

    # Sweep over max_new_tokens
    for max_nt in range(min_new_tokens, max_new_tokens + 1, step_new_tokens):
        # Warmup (not measured)
        for _ in range(warmup_iters):
            _ = run_once_dist(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=max_nt,
                rank=rank,
                world_size=world_size,
            )

        # Measured iterations (rank 0 collects metrics)
        prefill_vals = []
        ttft_vals = []
        tpot_vals = []
        total_vals = []
        decode_tps_vals = []
        e2e_tps_vals = []

        for _ in range(iterations):
            res = run_once_dist(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=max_nt,
                rank=rank,
                world_size=world_size,
            )
            if rank == 0 and res is not None:
                prefill_vals.append(res["prefill_ms"])
                ttft_vals.append(res["ttft_ms"])
                tpot_vals.append(res["avg_token_ms"])
                total_vals.append(res["total_ms"])
                decode_tps_vals.append(res["decode_tokens_per_s"])
                e2e_tps_vals.append(res["e2e_tokens_per_s"])

        if rank == 0:
            def mean(xs): return float(statistics.mean(xs)) if xs else 0.0
            def std(xs):  return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0

            prefill_mean = mean(prefill_vals)
            prefill_std = std(prefill_vals)
            ttft_mean = mean(ttft_vals)
            ttft_std = std(ttft_vals)
            tpot_mean = mean(tpot_vals)
            tpot_std = std(tpot_vals)
            total_mean = mean(total_vals)
            total_std = std(total_vals)
            decode_tps_mean = mean(decode_tps_vals)
            decode_tps_std = std(decode_tps_vals)
            e2e_tps_mean = mean(e2e_tps_vals)
            e2e_tps_std = std(e2e_tps_vals)

            print(
                f"{world_size},"
                f"{config.d_model},"
                f"{config.n_heads},"
                f"{config.n_layers},"
                f"{config.d_ff},"
                f"{config.vocab_size},"
                f"{prompt_len},"
                f"{max_nt},"
                f"{iterations},"
                f"{warmup_iters},"
                f"{prefill_mean:.3f},"
                f"{prefill_std:.3f},"
                f"{ttft_mean:.3f},"
                f"{ttft_std:.3f},"
                f"{tpot_mean:.3f},"
                f"{tpot_std:.3f},"
                f"{total_mean:.3f},"
                f"{total_std:.3f},"
                f"{decode_tps_mean:.3f},"
                f"{decode_tps_std:.3f},"
                f"{e2e_tps_mean:.3f},"
                f"{e2e_tps_std:.3f}"
            )


# ----------------------------
# CLI / main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Distributed tensor-parallel GPT inference benchmark (random weights)"
    )
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--max-seq-len", type=int, default=512)

    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--warmup-iters", type=int, default=3)

    p.add_argument("--min-new-tokens", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--step-new-tokens", type=int, default=8)

    return p.parse_args()


def main():
    args = parse_args()

    # Initialize distributed
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Select device based on LOCAL_RANK (torchrun sets this)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"# world_size={world_size}, backend={backend}, local_rank={local_rank}")

    config = DistGPTConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
    )

    # Sanity checks for TP
    assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
    assert config.n_heads % world_size == 0, "n_heads must be divisible by world_size"
    assert config.d_ff % world_size == 0, "d_ff must be divisible by world_size"
    assert args.prompt_len <= config.max_seq_len, "prompt_len > max_seq_len"

    run_benchmark_dist(
        config=config,
        iterations=args.iterations,
        warmup_iters=args.warmup_iters,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        step_new_tokens=args.step_new_tokens,
        prompt_len=args.prompt_len,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()