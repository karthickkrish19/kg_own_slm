#!/usr/bin/env python3
"""
SLM Production Suite — Unified Entry Point
===========================================
Train, generate, serve API, verify, and split data — all from one command.

Usage:
  python main.py split                                    # split corpus
  python main.py train --config config.yaml               # train model
  python main.py generate --checkpoint out/checkpoints/best.pt --prompt "Hello"
  python main.py stream   --checkpoint out/checkpoints/best.pt --prompt "Once upon"
  python main.py api      --checkpoint out/checkpoints/best.pt
  python main.py verify                                   # run all checks
  python main.py info     --checkpoint out/checkpoints/best.pt  # model info
"""

import os
import sys
import math
import argparse
import logging
from pathlib import Path

os.makedirs("out", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("out/main.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLM Production Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── split ─────────────────────────────────────────────────────────────────
    sp = sub.add_parser("split", help="Split corpus into train/val/test")
    sp.add_argument("--input", default="data/input.txt", help="input .txt file")
    sp.add_argument("--output_dir", default="data", help="output directory")
    sp.add_argument("--train", type=float, default=0.80, help="train ratio")
    sp.add_argument("--val", type=float, default=0.10, help="val ratio")
    sp.add_argument("--test", type=float, default=0.10, help="test ratio")
    sp.add_argument("--seed", type=int, default=42, help="random seed")
    sp.add_argument("--no_shuffle", action="store_true", help="disable shuffle")
    sp.add_argument("--dedupe", action="store_true", help="deduplicate chunks")
    sp.add_argument(
        "--mode",
        choices=["auto", "paragraph", "line", "character"],
        default="auto",
        help="split mode",
    )
    sp.add_argument("--config", default="config.yaml", help="config.yaml path")

    # ── train ─────────────────────────────────────────────────────────────────
    tp = sub.add_parser("train", help="Train the SLM model")
    tp.add_argument("--config", default="config.yaml", help="config file")
    tp.add_argument("--epochs", type=int, help="override epochs")
    tp.add_argument("--lr", type=float, help="override learning rate")
    tp.add_argument("--batch", type=int, help="override batch size")
    tp.add_argument("--resume", action="store_true", help="resume from checkpoint")
    tp.add_argument("--fresh", action="store_true", help="retrain tokenizer")
    tp.add_argument("--device", type=str, help="force device: cuda / cpu / mps")
    tp.add_argument("--compile", action="store_true", help="torch.compile")
    tp.add_argument("--no-wandb", action="store_true", help="disable W&B")
    tp.add_argument("--no-tb", action="store_true", help="disable TensorBoard")

    # ── generate ──────────────────────────────────────────────────────────────
    gp = sub.add_parser("generate", help="Generate text from prompt")
    gp.add_argument("--config", default="config.yaml", help="config file")
    gp.add_argument("--checkpoint", required=True, help="model checkpoint")
    gp.add_argument("--prompt", required=True, help="input prompt")
    gp.add_argument("--max_new_tokens", type=int, help="max tokens to generate")
    gp.add_argument("--temperature", type=float, help="sampling temperature")
    gp.add_argument("--top_k", type=int, help="top-k sampling")
    gp.add_argument("--top_p", type=float, help="nucleus sampling")
    gp.add_argument("--device", type=str, help="force device")

    # ── stream ────────────────────────────────────────────────────────────────
    stp = sub.add_parser("stream", help="Stream text generation")
    stp.add_argument("--config", default="config.yaml", help="config file")
    stp.add_argument("--checkpoint", required=True, help="model checkpoint")
    stp.add_argument("--prompt", required=True, help="input prompt")
    stp.add_argument("--max_new_tokens", type=int, help="max tokens")
    stp.add_argument("--temperature", type=float, help="temperature")
    stp.add_argument("--device", type=str, help="force device")

    # ── api ───────────────────────────────────────────────────────────────────
    ap = sub.add_parser("api", help="Start FastAPI server")
    ap.add_argument("--config", default="config.yaml", help="config file")
    ap.add_argument("--checkpoint", required=True, help="model checkpoint")

    # ── verify ────────────────────────────────────────────────────────────────
    sub.add_parser("verify", help="Run verification checks")

    # ── info ──────────────────────────────────────────────────────────────────
    ip = sub.add_parser("info", help="Show model info")
    ip.add_argument("--checkpoint", required=True, help="model checkpoint")

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────
def cmd_split(args) -> None:
    from split_data import split_data

    split_data(
        input_file=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        dedupe=args.dedupe,
        mode=args.mode,
    )


def cmd_train(args) -> None:
    # Import and run the training module
    # Build argv list for train.parse_args without mutating global sys.argv
    train_argv = []
    if args.config:
        train_argv.extend(["--config", args.config])
    if args.epochs:
        train_argv.extend(["--epochs", str(args.epochs)])
    if args.lr:
        train_argv.extend(["--lr", str(args.lr)])
    if args.batch:
        train_argv.extend(["--batch", str(args.batch)])
    if args.resume:
        train_argv.append("--resume")
    if args.fresh:
        train_argv.append("--fresh")
    if args.device:
        train_argv.extend(["--device", args.device])
    if getattr(args, "compile", False):
        train_argv.append("--compile")
    if getattr(args, "no_wandb", False):
        train_argv.append("--no-wandb")
    if getattr(args, "no_tb", False):
        train_argv.append("--no-tb")

    # Temporarily swap sys.argv, then restore
    saved_argv = sys.argv
    try:
        sys.argv = ["train.py"] + train_argv
        from train import main as train_main
        train_main()
    finally:
        sys.argv = saved_argv


def cmd_generate(args) -> None:
    from config import Config
    from inference import SLMInference

    cfg = (
        Config.from_yaml(args.config)
        if Path(args.config).exists()
        else Config()
    )

    engine = SLMInference(cfg, args.checkpoint, device=args.device)

    gen_kwargs = {}
    if args.max_new_tokens:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_k is not None:
        gen_kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p

    text = engine.generate(args.prompt, **gen_kwargs)
    print("\n" + "=" * 50)
    print("PROMPT:", args.prompt)
    print("=" * 50)
    print(text)
    print("=" * 50)


def cmd_stream(args) -> None:
    from config import Config
    from inference import SLMInference

    cfg = (
        Config.from_yaml(args.config)
        if Path(args.config).exists()
        else Config()
    )

    engine = SLMInference(cfg, args.checkpoint, device=args.device)

    gen_kwargs = {}
    if args.max_new_tokens:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature

    print("\n" + "=" * 50)
    print("PROMPT:", args.prompt)
    print("=" * 50)
    for chunk in engine.stream(args.prompt, **gen_kwargs):
        print(chunk, end="", flush=True)
    print("\n" + "=" * 50)


def cmd_api(args) -> None:
    from config import Config
    from api_server import run_api

    cfg = (
        Config.from_yaml(args.config)
        if Path(args.config).exists()
        else Config()
    )
    run_api(cfg, args.checkpoint)


def cmd_verify(args) -> None:
    import torch

    print("=" * 50)
    print("  SLM Production Verification")
    print("=" * 50)
    errors = []

    # 1. Config
    print("\n[1/7] Config...", end=" ")
    try:
        from config import Config

        cfg = Config()
        assert cfg.block_size > 0
        if Path("config.yaml").exists():
            cfg = Config.from_yaml("config.yaml")
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append(f"Config: {e}")

    # 2. Model
    print("[2/7] Model...", end=" ")
    test_cfg = None
    try:
        from model import SLM, SLMConfig

        test_cfg = SLMConfig(
            vocab_size=1000,
            max_seq_len=32,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=1,
        )
        model = SLM(test_cfg)
        x = torch.randint(1, 1000, (2, 16))
        logits, loss, acc = model(x, targets=x)
        assert logits.shape == (2, 16, 1000)
        assert loss is not None
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append(f"Model: {e}")

    # 3. Tokenizer
    print("[3/7] Tokenizer...", end=" ")
    try:
        from tokenizer_utils import SLMTokenizer, TOKENIZERS_AVAILABLE

        assert TOKENIZERS_AVAILABLE, "tokenizers library not installed"
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append(f"Tokenizer: {e}")

    # 4. Dataset
    print("[4/7] Dataset...", end=" ")
    try:
        from dataset import TokenizedDataset, slm_collate_fn

        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append(f"Dataset: {e}")

    # 5. Generation (with dummy model)
    print("[5/7] Generation...", end=" ")
    try:
        if test_cfg is None:
            from model import SLM, SLMConfig
            test_cfg = SLMConfig(
                vocab_size=1000, max_seq_len=32, embed_dim=64,
                num_layers=2, num_heads=4, num_kv_heads=1,
            )
        model = SLM(test_cfg)
        idx = torch.randint(1, 1000, (1, 8))
        out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=10)
        assert out.shape[1] > 8
        # Stream test
        for tok_id in model.stream_generate(idx, max_new_tokens=3):
            assert isinstance(tok_id, int)
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append(f"Generation: {e}")

    # 6. Inference engine
    print("[6/7] Inference module...", end=" ")
    try:
        from inference import SLMInference

        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        errors.append(f"Inference: {e}")

    # 7. Data files
    print("[7/7] Data files...", end=" ")
    data_ok = True
    for f in ["data/input.txt"]:
        if not Path(f).exists():
            print(f"WARN: {f} missing")
            data_ok = False
    if data_ok:
        print("OK")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"  VERIFICATION FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
    else:
        print("  ALL CHECKS PASSED")
    print("=" * 50)

    if errors:
        sys.exit(1)


def cmd_info(args) -> None:
    import torch
    from model import SLM

    model = SLM.load(args.checkpoint)
    info = model.model_info()

    print("=" * 50)
    print("  MODEL INFO")
    print("=" * 50)
    for k, v in info.items():
        if isinstance(v, (int, float)):
            print(f"  {k:25s} = {v:,}" if isinstance(v, int) else f"  {k:25s} = {v}")
        else:
            print(f"  {k:25s} = {v}")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "split": cmd_split,
        "train": cmd_train,
        "generate": cmd_generate,
        "stream": cmd_stream,
        "api": cmd_api,
        "verify": cmd_verify,
        "info": cmd_info,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
