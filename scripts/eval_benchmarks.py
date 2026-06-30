"""Evaluate a NewTale checkpoint on standard LLM benchmarks via lm-evaluation-harness.

Usage:
    pip install lm-eval
    python scripts/eval_benchmarks.py \\
        --config configs/3b.yaml \\
        --checkpoint checkpoints/3b/checkpoint-best \\
        --tasks hellaswag,arc_easy,arc_challenge,winogrande,mmlu \\
        --output results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from config import load_config
from model.transformer import NewTaleForCausalLM
from tokenizer.tokenizer import NewTaleTokenizer
from training.checkpoint import CheckpointManager


def _load_model(
    config_path: str, checkpoint_path: str
) -> tuple[NewTaleForCausalLM, NewTaleTokenizer]:
    config = load_config(config_path)
    tokenizer = NewTaleTokenizer(config.data.tokenizer_dir)
    config.model.vocab_size = tokenizer.vocab_size

    model = NewTaleForCausalLM(config.model)
    ckpt = CheckpointManager(Path(checkpoint_path).parent)
    ckpt.load_weights(checkpoint_path, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, tokenizer


class NewTaleLMAdapter:
    """Adapter implementing the lm_eval.api.model.LM interface."""

    def __init__(self, model: NewTaleForCausalLM, tokenizer: NewTaleTokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = next(model.parameters()).device
        self._max_length = model.config.max_position_embeddings

    # --- required lm_eval interface methods ---

    def tok_encode(self, string: str) -> list[int]:
        return self._tokenizer.encode(string)

    def tok_decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    @torch.no_grad()
    def loglikelihood(
        self, requests: list[tuple[str, str]]
    ) -> list[tuple[float, bool]]:
        results = []
        for context, continuation in requests:
            ctx_ids = self._tokenizer.encode(context)
            cont_ids = self._tokenizer.encode(continuation)
            all_ids = (ctx_ids + cont_ids)[: self._max_length]
            input_tensor = torch.tensor([all_ids], device=self._device)
            labels = input_tensor.clone()
            # Only compute loss over the continuation tokens
            labels[0, : len(ctx_ids)] = -100
            loss, _ = self._model(input_tensor, labels=labels)
            assert loss is not None
            ll = -loss.item() * len(cont_ids)
            # Greedy "is_greedy" flag — whether the model's argmax matches continuation
            logits_out, _ = self._model(input_tensor)
            greedy = logits_out[0, len(ctx_ids) - 1 : len(ctx_ids) - 1 + len(cont_ids)]
            is_greedy = all(
                greedy[i].argmax().item() == cont_ids[i] for i in range(len(cont_ids))
            )
            results.append((ll, is_greedy))
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[str]) -> list[float]:
        results = []
        for text in requests:
            ids = self._tokenizer.encode(text)[: self._max_length]
            input_tensor = torch.tensor([ids], device=self._device)
            loss, _ = self._model(input_tensor, labels=input_tensor.clone())
            assert loss is not None
            results.append(-loss.item() * len(ids))
        return results

    @torch.no_grad()
    def generate_until(self, requests: list[tuple[str, dict[str, Any]]]) -> list[str]:
        results = []
        for context, gen_kwargs in requests:
            ids = self._tokenizer.encode(context)[: self._max_length - 1]
            max_new = gen_kwargs.get("max_new_tokens", 64)
            generated = list(ids)
            for _ in range(max_new):
                inp = torch.tensor(
                    [generated[-self._max_length :]], device=self._device
                )
                _, logits = self._model(inp)
                next_tok = logits[0, -1].argmax().item()
                generated.append(int(next_tok))
                if int(next_tok) == self._tokenizer.eos_id:
                    break
            results.append(self._tokenizer.decode(generated[len(ids) :]))
        return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--tasks",
        default="hellaswag,arc_easy,arc_challenge,winogrande",
        help="Comma-separated lm-eval task names",
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    try:
        import lm_eval  # type: ignore[import-untyped]
    except ImportError:
        raise SystemExit("lm-eval not installed. Run: pip install lm-eval") from None

    model, tokenizer = _load_model(args.config, args.checkpoint)
    adapter = NewTaleLMAdapter(model, tokenizer)

    results = lm_eval.evaluator.evaluate(  # type: ignore[attr-defined]
        lm=adapter,
        task_manager=lm_eval.tasks.TaskManager(),  # type: ignore[attr-defined]
        task_dict=lm_eval.tasks.get_task_dict(args.tasks.split(",")),  # type: ignore[attr-defined]
        num_fewshot=args.num_fewshot,
        verbosity="INFO",
    )

    Path(args.output).write_text(json.dumps(results, indent=2, default=str))
    print(f"Results written to {args.output}")

    for task, res in results.get("results", {}).items():
        acc = (
            res.get("acc,none")
            or res.get("acc_norm,none")
            or res.get("word_perplexity,none")
        )
        print(f"  {task}: {acc}")


if __name__ == "__main__":
    main()
