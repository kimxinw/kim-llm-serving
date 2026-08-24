from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .api import create_app
from .config import GatewayConfig
from .tokenizer import HuggingFaceTokenizer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible Gateway for kim-llm-serving"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="run the HTTP/SSE Gateway")
    serve.add_argument("--config", required=True, type=Path)

    inspect = subparsers.add_parser(
        "inspect-tokenizer",
        help="print the fixed Tokenizer fingerprint and special Token IDs",
    )
    inspect.add_argument("--tokenizer-path", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect-tokenizer":
        tokenizer = HuggingFaceTokenizer.load(args.tokenizer_path)
        print(
            json.dumps(
                {
                    "tokenizer_fingerprint": tokenizer.tokenizer_fingerprint,
                    "chat_template_fingerprint": (
                        tokenizer.chat_template_fingerprint
                    ),
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    config = GatewayConfig.load(args.config)
    try:
        import uvicorn
    except ImportError as exception:
        raise RuntimeError("uvicorn is required to serve the Gateway") from exception
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
