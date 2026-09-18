"""Tokenizer wrapper used by the Qwen3 implementation.

Adapted from Sebastian Raschka's reasoning-from-scratch project.
"""

from pathlib import Path
import re

from tokenizers import Tokenizer


class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(
        self,
        tokenizer_file_path: str | Path = "tokenizer-base.json",
        apply_chat_template: bool = False,
        add_generation_prompt: bool = False,
        add_thinking: bool = False,
    ) -> None:
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tokenizer_path = Path(tokenizer_file_path)
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer file '{tokenizer_path}' not found")
        self._tok = Tokenizer.from_file(str(tokenizer_path))
        self._special_to_id = {
            token: self._tok.token_to_id(token) for token in self._SPECIALS
        }
        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id[self.pad_token]

        filename = tokenizer_path.name.lower()
        self.eos_token = (
            "<|endoftext|>"
            if "base" in filename and "reasoning" not in filename
            else "<|im_end|>"
        )
        self.eos_token_id = self._special_to_id[self.eos_token]

    def encode(self, prompt: str, chat_wrapped: bool | None = None) -> list[int]:
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template
        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]
        if chat_wrapped:
            prompt = self._wrap_chat(prompt)

        token_ids: list[int] = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                token_ids.append(self._special_to_id[part])
            else:
                token_ids.extend(self._tok.encode(part).ids)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_message: str) -> str:
        prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n"
        if self.add_generation_prompt:
            prompt += "<|im_start|>assistant"
            prompt += "\n" if self.add_thinking else "\n<think>\n\n</think>\n\n"
        return prompt
