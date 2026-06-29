from __future__ import annotations

from pathlib import Path

from transformers import PreTrainedTokenizerFast  # type: ignore[import-untyped]


class NewTaleTokenizer:
    def __init__(self, tokenizer_dir: str | Path) -> None:
        tokenizer_dir = Path(tokenizer_dir)
        self._tok = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )

    @property
    def bos_id(self) -> int:
        return self._tok.bos_token_id  # type: ignore[return-value]

    @property
    def eos_id(self) -> int:
        return self._tok.eos_token_id  # type: ignore[return-value]

    @property
    def pad_id(self) -> int:
        return self._tok.pad_token_id  # type: ignore[return-value]

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        result = self._tok.decode(ids, skip_special_tokens=False)
        return result if isinstance(result, str) else result[0]
