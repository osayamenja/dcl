import argparse
import time
import statistics
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
import nvtx
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFInferenceServer:
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
    ):
        """
        model_name: Hugging Face model id, e.g.:
            - "gpt2" for quick tests
            - "meta-llama/Meta-Llama-3-8B-Instruct" (if you have the weights & VRAM)
        device: "cuda" or "cpu". If None, auto-detect.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)


        print(f"# Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad_token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"# Loading model for {model_name} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

        self.max_seq_len = getattr(self.model.config, "max_position_embeddings", 2048)

        torch.manual_seed(0)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(0)

        print("# Model and tokenizer ready.")

    @nvtx.annotate(color="blue")
    @torch.no_grad()
    def run_once(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        return_text: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a single prefill + decode pass and return metrics.

        Returns:
          - 'prefill_ms'
          - 'ttft_ms'
          - 'avg_token_ms'
          - 'token_latencies_ms'
          - 'total_ms'
          - 'decode_tokens_per_s'
          - 'e2e_tokens_per_s'
          - 'output_text' (optional, only if return_text=True)
        """
        # Encode prompt
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)        # (1, T)
        attention_mask = enc["attention_mask"].to(self.device)

        # Truncate to context window from the right
        if input_ids.size(1) > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len :]
            attention_mask = attention_mask[:, -self.max_seq_len :]

        use_cuda_timing = self.device.type == "cuda" and torch.cuda.is_available()

        if use_cuda_timing:
            torch.cuda.synchronize()

        total_start = time.perf_counter()

        # -------------------
        # Prefill
        # -------------------
        if use_cuda_timing:
            prefill_start_evt = torch.cuda.Event(enable_timing=True)
            prefill_end_evt = torch.cuda.Event(enable_timing=True)
            prefill_start_evt.record()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

            prefill_end_evt.record()
            torch.cuda.synchronize()
            prefill_ms = prefill_start_evt.elapsed_time(prefill_end_evt)
        else:
            t0 = time.perf_counter()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            t1 = time.perf_counter()
            prefill_ms = (t1 - t0) * 1000.0

        logits = outputs.logits               # (1, T, V)
        past_key_values = outputs.past_key_values

        # For decode step 0:
        #   - Input is the last prompt token
        #   - We time this step as part of decode loop
        last_prompt_token = input_ids[:, -1:]  # shape (1, 1)

        token_latencies_ms: List[float] = []
        generated_ids: List[int] = []

        current_input = last_prompt_token

        # Decode loop
        for step in range(max_new_tokens):
            if use_cuda_timing:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()

                out = self.model(
                    input_ids=current_input,
                    use_cache=True,
                    past_key_values=past_key_values,
                )

                end_evt.record()
                torch.cuda.synchronize()
                step_ms = start_evt.elapsed_time(end_evt)
            else:
                t0 = time.perf_counter()
                out = self.model(
                    input_ids=current_input,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                t1 = time.perf_counter()
                step_ms = (t1 - t0) * 1000.0

            token_latencies_ms.append(step_ms)

            logits_step = out.logits  # (1, 1, V)
            past_key_values = out.past_key_values

            logits_last = logits_step[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                k = min(top_k, logits_last.size(-1))
                v, _ = torch.topk(logits_last, k)
                thresh = v[:, -1].unsqueeze(-1)
                logits_last = torch.where(
                    logits_last < thresh,
                    torch.full_like(logits_last, float("-inf")),
                    logits_last,
                )

            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            generated_ids.append(next_token.item())
            current_input = next_token

        total_ms = (time.perf_counter() - total_start) * 1000.0

        # Metrics
        ttft_ms = prefill_ms + (token_latencies_ms[0] if token_latencies_ms else 0.0)
        avg_token_ms = (
            sum(token_latencies_ms) / len(token_latencies_ms)
            if token_latencies_ms
            else 0.0
        )

        # Throughputs
        total_decode_ms = sum(token_latencies_ms) if token_latencies_ms else 0.0
        total_decode_s = total_decode_ms / 1000.0 if total_decode_ms > 0.0 else float("inf")
        total_e2e_s = total_ms / 1000.0 if total_ms > 0.0 else float("inf")

        decode_tokens_per_s = (
            max_new_tokens / total_decode_s if total_decode_s > 0.0 else 0.0
        )
        e2e_tokens_per_s = (
            max_new_tokens / total_e2e_s if total_e2e_s > 0.0 else 0.0
        )

        result: Dict[str, Any] = {
            "prefill_ms": prefill_ms,
            "ttft_ms": ttft_ms,
            "avg_token_ms": avg_token_ms,
            "token_latencies_ms": token_latencies_ms,
            "total_ms": total_ms,
            "decode_tokens_per_s": decode_tokens_per_s,
            "e2e_tokens_per_s": e2e_tokens_per_s,
        }

        if return_text:
            text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            result["output_text"] = text

        return result


def run_benchmark(
    model_name: str,
    device: Optional[str],
    prompt: str,
    iterations: int,
    warmup_iters: int,
    min_new_tokens: int,
    max_new_tokens: int,
    step_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    print_text: bool,
):
    server = HFInferenceServer(model_name=model_name, device=device)

    # Encode once to report prompt length
    tok = server.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    prompt_len = tok["input_ids"].shape[1]

    # CSV header
    # One row per max_new_tokens configuration
    print(
        "model,"
        "device,"
        "prompt_len,"
        "max_new_tokens,"
        "iterations,"
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

    device_str = str(server.device)

    max_nt = min_new_tokens
    while max_nt <= max_new_tokens:
        prefill_vals = []
        ttft_vals = []
        tpot_vals = []
        total_vals = []
        decode_tps_vals = []
        e2e_tps_vals = []
        sample_text = None

        for _ in range(warmup_iters):
            server.run_once(
                prompt=prompt,
                max_new_tokens=max_nt,
                temperature=temperature,
                top_k=top_k,
                return_text=False,
            )

        for it in range(iterations):
            res = server.run_once(
                prompt=prompt,
                max_new_tokens=max_nt,
                temperature=temperature,
                top_k=top_k,
                return_text=(print_text and it == iterations - 1),
            )
            prefill_vals.append(res["prefill_ms"])
            ttft_vals.append(res["ttft_ms"])
            tpot_vals.append(res["avg_token_ms"])
            total_vals.append(res["total_ms"])
            decode_tps_vals.append(res["decode_tokens_per_s"])
            e2e_tps_vals.append(res["e2e_tokens_per_s"])

            if print_text and "output_text" in res:
                sample_text = res["output_text"]

        # Stats helpers
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

        # CSV row
        print(
            f"{model_name},"
            f"{device_str},"
            f"{prompt_len},"
            f"{max_nt},"
            f"{iterations},"
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

        if print_text and sample_text is not None:
            print(f"# Sample output for max_new_tokens={max_nt}:")
            print(f"PROMPT: {prompt}")
            print(f"OUTPUT: {sample_text}")
            print("# -----")
        max_nt = int(max_nt * step_new_tokens)


def parse_args():
    p = argparse.ArgumentParser(description="HF single-GPU inference benchmark harness")
    p.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HF model name or path (default: Llama-3.1-8B-Instruct)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device, e.g. "cuda" or "cpu" (default: auto-detect)',
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumped over the lazy dog. This is a test prompt for measuring TTFT and TPOT.",
        help="Prompt text to use for benchmarking",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=16,
        help="Number of iterations per max_new_tokens setting",
    )
    p.add_argument(
        "--warmup-iterations",
        type=int,
        default=8,
        help="Number of warmup iterations per max_new_tokens setting",
    )
    p.add_argument(
        "--min-new-tokens",
        type=int,
        default=8,
        help="Minimum number of generated tokens in the sweep",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens in the sweep",
    )
    p.add_argument(
        "--step-new-tokens",
        type=int,
        default=2,
        help="Geometric Step size for generated tokens in the sweep",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (set <=0 to disable)",
    )
    p.add_argument(
        "--print-text",
        action="store_true",
        help="Print a sample generated text per max_new_tokens setting",
    )
    return p.parse_args()


def main():
    args = parse_args()

    top_k = args.top_k if args.top_k > 0 else None

    run_benchmark(
        model_name=args.model_name,
        device=args.device,
        prompt=args.prompt,
        iterations=args.iterations,
        warmup_iters=args.warmup_iterations,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        step_new_tokens=args.step_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        print_text=args.print_text,
    )


if __name__ == "__main__":
    main()