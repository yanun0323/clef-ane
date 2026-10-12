"""Small command-line entrypoint; --help and downloads never load the ANE or Torch."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, paths
from .bundle import BUCKETS
from .models import DEFAULT_MODEL, MODELS


def port_number(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="clef-ane", description="Run Clef and Clef-Flash decisions on the Apple Neural Engine.")
    result.add_argument("--version", action="version", version=f"clef-ane {__version__}")
    commands = result.add_subparsers(dest="command", required=True)
    model_help = "which model: clef-flash (9B, faster) or clef (27B, larger; needs about 23 GiB of disk) (default: %(default)s)"
    directory_help = f"model directory (default: a folder per model in {paths.home() / 'models'})"
    download = commands.add_parser("download", help="download and check a ready-to-run LUT-6 model")
    download.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL, help=model_help)
    download.add_argument("--repo", help="Hugging Face model repository (OWNER/NAME); remembered for later downloads")
    download.add_argument("--revision", default="main", help="Hub branch, tag, or commit (default: main)")
    download.add_argument("--model-dir", type=Path, help=directory_help)
    download.add_argument("--force", action="store_true", help="download files again, including damaged cached files")
    serve = commands.add_parser("serve", help="start the local decision API; never downloads automatically")
    serve.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL, help=model_help)
    serve.add_argument("--model-dir", type=Path, help=directory_help)
    serve.add_argument("--host", default="127.0.0.1", help="listen address (default: %(default)s); there is no authentication")
    serve.add_argument("--port", type=port_number, default=8080, help="listen port (default: %(default)s)")
    serve.add_argument("--max-tokens", type=int, choices=BUCKETS, default=BUCKETS[-1], help="maximum input length (default: %(default)s)")
    serve.add_argument("--max-body-mb", type=positive_int, default=32, help="maximum JSON body size in MiB, including base64 overhead (default: %(default)s)")
    serve.add_argument("--name", help="model name reported by the API (default: the model's own name)")
    serve.add_argument("--vision-device", choices=("ane", "mps", "cpu"), default="ane",
                       help="vision encoder device: ane keeps the GPU idle (default), mps uses the Metal GPU (faster), "
                            "cpu is slowest; media bundles only")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    model_dir = args.model_dir or paths.model_dir(args.model)
    try:
        if args.command == "download":
            from .download import download_model
            download_model(model_dir, args.repo, args.revision, force=args.force, default_repo=MODELS[args.model].repo_id)
        else:
            from .server import serve
            serve(model_dir, host=args.host, port=args.port, max_tokens=args.max_tokens,
                  name=args.name, max_body_mb=args.max_body_mb, vision_device=args.vision_device)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        from filelock import Timeout
        if isinstance(error, Timeout):
            message = "This model is already being served or downloaded. Stop that process, or use another --model-dir."
        else:
            message = str(error)
        print(f"clef-ane: {message}", file=sys.stderr)
        return 1
    return 0
