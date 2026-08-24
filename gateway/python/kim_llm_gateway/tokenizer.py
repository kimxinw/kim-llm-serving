from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from kim_llm_client import ModelManifest


TOKENIZER_FINGERPRINT_VERSION = b"kim-llm-tokenizer-fingerprint-v1\0"
CHAT_TEMPLATE_FINGERPRINT_VERSION = b"kim-llm-chat-template-v1\0"
TOKENIZER_ASSET_NAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


class TokenizerError(RuntimeError):
    pass


class Tokenizer(Protocol):
    @property
    def eos_token_id(self) -> int:
        ...

    @property
    def pad_token_id(self) -> int:
        ...

    @property
    def tokenizer_fingerprint(self) -> str:
        ...

    @property
    def chat_template_fingerprint(self) -> str:
        ...

    def encode_chat(self, messages: Sequence[Mapping[str, str]]) -> tuple[int, ...]:
        ...

    def encode_text(self, text: str) -> tuple[int, ...]:
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        ...


def fingerprint_tokenizer_assets(tokenizer_path: str | Path) -> str:
    root = Path(tokenizer_path).expanduser().resolve()
    if not root.is_dir():
        raise TokenizerError(f"Tokenizer directory does not exist: {root}")
    assets = [root / name for name in TOKENIZER_ASSET_NAMES if (root / name).is_file()]
    if not any(
        path.name in {"tokenizer.json", "tokenizer.model", "spiece.model"}
        for path in assets
    ):
        raise TokenizerError(
            "Tokenizer directory has no tokenizer.json, tokenizer.model, or spiece.model"
        )

    digest = hashlib.sha256()
    digest.update(TOKENIZER_FINGERPRINT_VERSION)
    for path in assets:
        name = path.name.encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as input_file:
            while True:
                chunk = input_file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def fingerprint_chat_template(chat_template: str) -> str:
    if not chat_template:
        raise TokenizerError("Tokenizer does not define a chat template")
    digest = hashlib.sha256()
    digest.update(CHAT_TEMPLATE_FINGERPRINT_VERSION)
    digest.update(chat_template.encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


class HuggingFaceTokenizer:
    def __init__(self, tokenizer: Any, tokenizer_path: Path) -> None:
        self._tokenizer = tokenizer
        self._tokenizer_path = tokenizer_path
        eos_token_id = tokenizer.eos_token_id
        if type(eos_token_id) is not int or eos_token_id < 0:
            raise TokenizerError("Tokenizer eos_token_id is missing or invalid")
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise TokenizerError("Tokenizer pad_token_id is invalid")
        chat_template = tokenizer.chat_template
        if not isinstance(chat_template, str) or not chat_template:
            raise TokenizerError("Tokenizer chat_template is missing")

        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        self._tokenizer_fingerprint = fingerprint_tokenizer_assets(tokenizer_path)
        self._chat_template_fingerprint = fingerprint_chat_template(chat_template)

    @classmethod
    def load(cls, tokenizer_path: str | Path) -> "HuggingFaceTokenizer":
        path = Path(tokenizer_path).expanduser().resolve()
        try:
            from transformers import AutoTokenizer
        except ImportError as exception:
            raise TokenizerError(
                "transformers is required to load the Gateway tokenizer"
            ) from exception
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
        except Exception as exception:
            raise TokenizerError(
                f"failed to load Tokenizer from {path}: {exception}"
            ) from exception
        return cls(tokenizer, path)

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def tokenizer_fingerprint(self) -> str:
        return self._tokenizer_fingerprint

    @property
    def chat_template_fingerprint(self) -> str:
        return self._chat_template_fingerprint

    def validate_manifest(self, manifest: ModelManifest) -> None:
        mismatches: list[str] = []
        comparisons = (
            ("tokenizer_fingerprint", self.tokenizer_fingerprint, manifest.tokenizer_fingerprint),
            (
                "chat_template_fingerprint",
                self.chat_template_fingerprint,
                manifest.chat_template_fingerprint,
            ),
            ("eos_token_id", self.eos_token_id, manifest.eos_token_id),
            ("pad_token_id", self.pad_token_id, manifest.pad_token_id),
        )
        for name, actual, expected in comparisons:
            if actual != expected:
                mismatches.append(f"{name}: expected {expected!r}, got {actual!r}")
        if mismatches:
            raise TokenizerError("Tokenizer/Worker manifest mismatch: " + "; ".join(mismatches))

    def encode_chat(self, messages: Sequence[Mapping[str, str]]) -> tuple[int, ...]:
        try:
            token_ids = self._tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
        except Exception as exception:
            raise TokenizerError(f"failed to apply chat template: {exception}") from exception
        return self._normalize_token_ids(token_ids, "chat template")

    def encode_text(self, text: str) -> tuple[int, ...]:
        try:
            token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        except Exception as exception:
            raise TokenizerError(f"failed to tokenize stop sequence: {exception}") from exception
        return self._normalize_token_ids(token_ids, "text")

    def decode(self, token_ids: Sequence[int]) -> str:
        try:
            return str(
                self._tokenizer.decode(
                    list(token_ids),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        except Exception as exception:
            raise TokenizerError(f"failed to decode generated tokens: {exception}") from exception

    @staticmethod
    def _normalize_token_ids(value: Any, description: str) -> tuple[int, ...]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if (
            not isinstance(value, (list, tuple))
            or not value
            or any(type(token_id) is not int or token_id < 0 for token_id in value)
        ):
            raise TokenizerError(f"{description} produced invalid Token IDs")
        return tuple(value)


__all__ = [
    "HuggingFaceTokenizer",
    "Tokenizer",
    "TokenizerError",
    "fingerprint_chat_template",
    "fingerprint_tokenizer_assets",
]
