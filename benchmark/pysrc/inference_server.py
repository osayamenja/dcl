import time
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
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

        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Make sure we have a pad token if needed
        if self.tokenizer.pad_token is None:
            # Common pattern for GPT-style models
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model for {model_name} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

        # For context window
        self.max_seq_len = getattr(self.model.config, "max_position_embeddings", 2048)

        # Seeds
        torch.manual_seed(0)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(0)

        print("Model and tokenizer ready.")

    @torch.no_grad()
    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 32,
            temperature: float = 1.0,
            top_k: Optional[int] = 50,
    ) -> Dict[str, Any]:
        """
        Manual prefill + decode loop with timing.

        Returns:
          - 'prompt'
          - 'output_text'
          - 'ttft_ms'
          - 'avg_token_ms'
          - 'token_latencies_ms' (list)
          - 'prefill_ms'
          - 'total_time_ms'
        """
        # -------------------
        # Encode prompt
        # -------------------
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self.device)  # (1, T)
        attention_mask = enc["attention_mask"].to(self.device)

        # Truncate to context window from the right
        if input_ids.size(1) > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len :]
            attention_mask = attention_mask[:, -self.max_seq_len :]

        # For decoding, we'll track only newly generated tokens
        generated_ids: List[int] = []

        use_cuda_timing = self.device.type == "cuda" and torch.cuda.is_available()

        if use_cuda_timing:
            torch.cuda.synchronize()

        total_start_cpu = time.perf_counter()

        # -------------------
        # Prefill phase
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

        logits = outputs.logits  # (1, T, V)
        past_key_values = outputs.past_key_values

        # Logits for last prompt token -> first sampled token
        last_logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            k = min(top_k, last_logits.size(-1))
            v, _ = torch.topk(last_logits, k)
            thresh = v[:, -1].unsqueeze(-1)
            last_logits = torch.where(
                last_logits < thresh,
                torch.full_like(last_logits, float("-inf")),
                last_logits,
                )

        probs = F.softmax(last_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        generated_ids.append(next_token_id.item())

        # -------------------
        # Decode loop with timing
        # -------------------
        token_latencies_ms: List[float] = []

        # First decode step was just sampled; now we loop including it and the rest,
        # but we want to time the forward passes, so we'll re-run them in a uniform loop.
        # For TTFT we define:
        #   TTFT = prefill_ms + first_decode_step_ms
        #
        # So we time max_new_tokens decode steps, starting from the first sampled token.

        # We'll re-use the sampled token as the first input token
        current_token = next_token_id  # (1, 1)

        for step in range(max_new_tokens):
            # Shape: (1, 1)
            step_input_ids = current_token.to(self.device)

            if use_cuda_timing:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()

                out = self.model(
                    input_ids=step_input_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                )

                end_evt.record()
                torch.cuda.synchronize()
                step_ms = start_evt.elapsed_time(end_evt)
            else:
                t0 = time.perf_counter()
                out = self.model(
                    input_ids=step_input_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                t1 = time.perf_counter()
                step_ms = (t1 - t0) * 1000.0

            token_latencies_ms.append(step_ms)

            logits = out.logits  # (1, 1, V)
            past_key_values = out.past_key_values

            logits_last = logits[:, -1, :] / max(temperature, 1e-5)

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

            # Save this token as generated (except that for the very first iteration
            # we've already counted the first sample once above; here we just append).
            generated_ids.append(next_token.item())

            # Next step's input
            current_token = next_token

        total_time_ms = (time.perf_counter() - total_start_cpu) * 1000.0

        ttft_ms = prefill_ms + (token_latencies_ms[0] if token_latencies_ms else 0.0)
        avg_token_ms = (
            sum(token_latencies_ms) / len(token_latencies_ms)
            if token_latencies_ms
            else None
        )

        # Decode only the newly generated tokens
        output_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return {
            "prompt": prompt,
            "output_text": output_text,
            "ttft_ms": ttft_ms,
            "avg_token_ms": avg_token_ms,
            "token_latencies_ms": token_latencies_ms,
            "prefill_ms": prefill_ms,
            "total_time_ms": total_time_ms,
        }


def main():
    # You can change this to a Llama-3.x model once you're ready, e.g.:
    #   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_name = "gpt2"
    server = HFInferenceServer(model_name=model_name)

    print("\nHF inference server ready. Type a prompt (or 'exit').\n")

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
        print(f"Prefill: {result['prefill_ms']:.3f} ms")
        print(f"TTFT:    {result['ttft_ms']:.3f} ms")
        print(f"TPOT:    {result['avg_token_ms']:.3f} ms (avg over {len(result['token_latencies_ms'])} tokens)")
        print(f"Total:   {result['total_time_ms']:.3f} ms (wall-clock)")
        print("-----------------\n")


if __name__ == "__main__":
    main()