"""Tokenizer wrapper for Qwen3.5.

Uses the HuggingFace tokenizer from the model directory.
Handles encoding, decoding, chat templates, and special tokens.
"""

from pathlib import Path
from typing import Optional

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class Tokenizer:
    """Wraps HuggingFace tokenizer for Qwen3.5.

    Usage:
        tok = Tokenizer.from_pretrained("mlx-community/Qwen3.5-397B-A17B-4bit")
        ids = tok.encode("Hello, world!")
        text = tok.decode(ids)
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = getattr(tokenizer, 'bos_token_id', None)
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', self.eos_token_id)
        self.vocab_size = tokenizer.vocab_size

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Tokenizer":
        """Load tokenizer from a HuggingFace model directory or repo."""
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for tokenizer. "
                "Install with: pip install transformers"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return cls(tokenizer)

    @classmethod
    def from_local(cls, model_dir: str) -> "Tokenizer":
        """Load tokenizer from a local directory."""
        return cls.from_pretrained(model_dir)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """Apply Qwen3.5 chat template to messages.

        Args:
            messages: List of {"role": "user"/"assistant"/"system", "content": "..."}
            add_generation_prompt: Whether to add the assistant prompt prefix

        Returns:
            Token IDs ready for inference
        """
        return self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )

    def format_prompt(self, user_message: str, system_prompt: Optional[str] = None) -> list[int]:
        """Convenience: format a single user message into token IDs."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        return self.apply_chat_template(messages)
