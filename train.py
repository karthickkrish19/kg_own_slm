import os
import sys
import math
import time
import logging
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.makedirs("out", exist_ok=True)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("out/train.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)
logger.info("\n" + "=" * 60 + "\n  NEW TRAINING RUN\n" + "=" * 60)

# ── Optional loggers ──────────────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    logger.info("TensorBoard not installed — pip install tensorboard")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.info("W&B not installed — pip install wandb")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SLM")
    p.add_argument("--config",    default="config.yaml", help="config file path")
    p.add_argument("--epochs",    type=int,   help="override epochs")
    p.add_argument("--lr",        type=float, help="override learning rate")
    p.add_argument("--batch",     type=int,   help="override batch size")
    p.add_argument("--resume",    action="store_true", help="resume from last checkpoint")
    p.add_argument("--fresh",     action="store_true", help="retrain tokenizer from scratch")
    p.add_argument("--device",    type=str,   help="force device: cuda / cpu / mps")
    p.add_argument("--compile",   action="store_true", help="torch.compile the model")
    p.add_argument("--no-wandb",  action="store_true", help="disable W&B")
    p.add_argument("--no-tb",     action="store_true", help="disable TensorBoard")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule — cosine decay with linear warmup
# ─────────────────────────────────────────────────────────────────────────────
def get_lr(
    step:         int,
    warmup_steps: int,
    total_steps:  int,
    max_lr:       float,
    min_lr:       float,
) -> float:
    
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


# ─────────────────────────────────────────────────────────────────────────────
# AMP helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_amp_dtype(
    device:           torch.device,
    mixed_precision:  str,
) -> torch.dtype:
    
    if device.type != "cuda":
        return torch.bfloat16
    return (
        torch.float16 if mixed_precision == "float16"
        else torch.bfloat16
    )


def _make_scaler(
    device:          torch.device,
    mixed_precision: str,
) -> torch.cuda.amp.GradScaler:
    
    enabled = (
        device.type == "cuda" and mixed_precision == "float16"
    )
    try:
        # PyTorch ≥ 2.4
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        # PyTorch < 2.4 fallback
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    amp_dtype:   torch.dtype,
    max_batches: int = 50,
) -> Tuple[float, float]:
    
    model.eval()
    total_loss = total_acc = count = 0

    with torch.no_grad():
        
        for i, (x, y, loss_mask) in enumerate(loader):
            if i >= max_batches:
                break

            x         = x.to(device,         non_blocking=True)
            y         = y.to(device,          non_blocking=True)
            loss_mask = loss_mask.to(device,  non_blocking=True)

            with torch.autocast(
                device_type = device.type,
                dtype       = amp_dtype,
                enabled     = (device.type != "cpu"),
            ):
                
                _, loss, acc = model(
                    x,
                    targets   = y,
                    loss_mask = loss_mask,
                )

            if loss is not None:
                total_loss += loss.item()
                total_acc  += acc
                count      += 1

    model.train()
    return (
        total_loss / max(count, 1),
        total_acc  / max(count, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_optimizer(
    model:        nn.Module,
    lr:           float,
    weight_decay: float,
    device:       torch.device,
) -> torch.optim.AdamW:
    
    decay_params    = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and "embed" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    logger.info(
        "Optimizer: decay_params=%d  no_decay_params=%d",
        len(decay_params), len(no_decay_params),
    )

    opt_kwargs = dict(
        lr    = lr,
        betas = (0.9, 0.95),
        eps   = 1e-8,
    )
    # fused AdamW is only available on CUDA with recent PyTorch
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(
                [
                    {"params": decay_params,    "weight_decay": weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                fused=True,
                **opt_kwargs,
            )
        except (TypeError, RuntimeError):
            logger.info("fused AdamW not available — using standard AdamW")

    return torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        **opt_kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Safe config dict
# ─────────────────────────────────────────────────────────────────────────────
def _safe_config_dict(model: nn.Module) -> dict:

    raw = getattr(model, "_orig_mod", model)
    cfg = getattr(raw, "config", None)
    if cfg is None:
        return {}
    from dataclasses import asdict
    try:
        return asdict(cfg)
    except Exception:
        return vars(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    from config import Config
    cfg = (
        Config.from_yaml(args.config)
        if Path(args.config).exists()
        else Config()
    )
    if args.epochs:  cfg.epochs        = args.epochs
    if args.lr:      cfg.learning_rate = args.lr
    if args.batch:   cfg.batch_size    = args.batch
    if args.compile: cfg.compile_model = True
    if args.no_wandb:cfg.use_wandb     = False
    if args.no_tb:   cfg.use_tensorboard = False

    # ── Directories ───────────────────────────────────────────────────────────
    os.makedirs(cfg.model_save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir,        exist_ok=True)
    os.makedirs("out",              exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    logger.info(
        "Device: %s | seed=%d | PyTorch=%s",
        device, cfg.seed, torch.__version__,
    )

    # ── AMP setup ─────────────────────────────────────────────────────────────
    amp_dtype = _get_amp_dtype(device, cfg.mixed_precision)
    scaler    = _make_scaler(device, cfg.mixed_precision) 
    use_amp   = device.type != "cpu"
    logger.info(
        "AMP: dtype=%s  scaler=%s",
        amp_dtype, use_amp,
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    from tokenizer_utils import SLMTokenizer, train_tokenizer
    tok_path   = cfg.tokenizer_file
    train_file = (
        cfg.corpus_files[0]
        if cfg.corpus_files
        else "data/train.txt"
    )

    # Use full corpus for tokenizer training (better vocab coverage)
    # Falls back to corpus_files if tokenizer_corpus_files not set
    tok_train_files = (
        cfg.tokenizer_corpus_files
        if getattr(cfg, "tokenizer_corpus_files", None)
        else cfg.corpus_files
    )

    if not Path(tok_path).exists() or args.fresh:
        logger.info("Training tokenizer from: %s", tok_train_files)
        tok = train_tokenizer(
            files         = tok_train_files,
            vocab_size    = cfg.vocab_size,
            save_path     = tok_path,
        )
    else:
        tok = SLMTokenizer(tok_path)
        logger.info(
            "Tokenizer loaded: %s  vocab=%d",
            tok_path, tok.vocab_size,
        )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    from dataset import get_dataloaders
    val_path = "data/val.txt"

    train_dl, val_dl, _ = get_dataloaders(
        train_file  = train_file,
        tokenizer   = tok,
        seq_len     = cfg.block_size,
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
        stride      = cfg.train_stride,
        val_file    = val_path if Path(val_path).exists() else None,
        seed        = cfg.seed,
        pad_id      = tok.pad_id,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    from model import SLM, SLMConfig
    model_cfg = SLMConfig(
        vocab_size             = tok.vocab_size,
        pad_id                 = tok.pad_id,
        max_seq_len            = cfg.block_size,
        embed_dim              = cfg.embed_dim,
        num_layers             = cfg.num_layers,
        num_heads              = cfg.num_heads,
        num_kv_heads           = cfg.num_kv_heads,
        dropout                = cfg.dropout,
        attn_dropout           = cfg.attn_dropout,
        ffn_dropout            = cfg.ffn_dropout,
        rope_base              = cfg.rope_base,
        cache_factor           = cfg.cache_factor,
        use_kv_cache           = cfg.use_kv_cache,
        tie_weights            = cfg.tie_weights,
        ffn_multiplier         = cfg.ffn_multiplier,
        multiple_of            = cfg.multiple_of,
        norm_eps               = cfg.norm_eps,
        init_scale             = cfg.init_scale,
        gradient_checkpointing = cfg.gradient_checkpointing,
        use_flash              = cfg.use_flash,
    )
    model = SLM(model_cfg).to(device)

    
    if cfg.compile_model and hasattr(torch, "compile"):
        logger.info("Applying torch.compile ...")
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %s params | %d layers | %d heads (%d kv)",
        f"{total_params:,}", cfg.num_layers,
        cfg.num_heads, cfg.num_kv_heads,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = _build_optimizer(
        model, cfg.learning_rate, cfg.weight_decay, device
    )

    # ── Training schedule ─────────────────────────────────────────────────────
    steps_per_epoch = len(train_dl)
    total_steps     = (
        cfg.epochs * steps_per_epoch // cfg.grad_accum_steps
    )
    logger.info(
        "Training plan: epochs=%d  steps_per_epoch=%d  "
        "grad_accum=%d  total_optimizer_steps=%d",
        cfg.epochs, steps_per_epoch,
        cfg.grad_accum_steps, total_steps,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step  = 0
    start_epoch  = 0
    best_loss    = float("inf")
    no_improve   = 0

    if args.resume:
        ckpts = sorted(
            Path(cfg.model_save_dir).glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if ckpts:
            ckpt_path = ckpts[-1]
            ckpt = torch.load(
                ckpt_path,
                map_location = device,
                weights_only = False,
            )
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            global_step = ckpt.get("step",       0)
            start_epoch = ckpt.get("epoch",      0)
            best_loss   = ckpt.get("best_loss",  float("inf"))
            no_improve  = ckpt.get("no_improve", 0)
            logger.info(
                "Resumed <- %s  (epoch=%d step=%d best_loss=%.4f)",
                ckpt_path, start_epoch, global_step, best_loss,
            )
        else:
            logger.warning(
                "No checkpoints found in %s — starting fresh.",
                cfg.model_save_dir,
            )

    # ── Loggers ───────────────────────────────────────────────────────────────
    writer = None
    if cfg.use_tensorboard and TB_AVAILABLE:
        writer = SummaryWriter(log_dir=cfg.log_dir)
        logger.info("TensorBoard -> %s", cfg.log_dir)

    if cfg.use_wandb and WANDB_AVAILABLE:
        
        run_id_file = Path(cfg.model_save_dir) / "wandb_run_id.txt"
        run_id = (
            run_id_file.read_text().strip()
            if run_id_file.exists() and args.resume
            else None
        )
        wandb.init(
            project = cfg.wandb_project,
            config  = vars(cfg),
            resume  = "allow",
            id      = run_id,
        )
        if run_id is None:
            run_id_file.write_text(wandb.run.id)

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    
    stop_training = False
    t0 = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs):
        if stop_training:
            break

        epoch_loss = 0.0
        epoch_acc  = 0.0
        n_batches  = 0

        
        for local_step, (x, y, loss_mask) in enumerate(train_dl):
            x         = x.to(device,        non_blocking=True)
            y         = y.to(device,         non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)

            # ── Forward ───────────────────────────────────────────────────────
            with torch.autocast(
                device_type = device.type,
                dtype       = amp_dtype,
                enabled     = use_amp,
            ):
                
                _, loss, acc = model(
                    x,
                    targets          = y,
                    loss_mask        = loss_mask,
                    label_smoothing  = cfg.label_smoothing,
                )
                scaled_loss = loss / cfg.grad_accum_steps

            scaler.scale(scaled_loss).backward()

            # ── Gradient accumulation step ────────────────────────────────────
            if (local_step + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # ── Update LR ─────────────────────────────────────────────────
                lr = get_lr(
                    global_step, cfg.warmup_steps, total_steps,
                    cfg.learning_rate, cfg.min_lr,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # ── Logging ───────────────────────────────────────────────────
                if global_step % cfg.log_every == 0:
                    elapsed  = time.perf_counter() - t0
                    tokens_s = (
                        cfg.log_every
                        * cfg.batch_size
                        * cfg.block_size
                        * cfg.grad_accum_steps
                    ) / max(elapsed, 1e-6)

                    
                    raw_loss = loss.item()
                    ppl      = math.exp(min(raw_loss, 88.0))

                    logger.info(
                        "epoch=%d  step=%d  loss=%.4f  ppl=%.1f  "
                        "acc=%.3f  lr=%.2e  grad_norm=%.3f  tok/s=%s",
                        epoch + 1, global_step, raw_loss,
                        ppl, acc, lr, grad_norm,
                        f"{tokens_s:,.0f}",
                    )

                    if writer:
                        writer.add_scalar("train/loss",           raw_loss,  global_step)
                        writer.add_scalar("train/ppl",            ppl,       global_step)
                        writer.add_scalar("train/acc",            acc,       global_step)
                        writer.add_scalar("train/lr",             lr,        global_step)
                        writer.add_scalar("train/grad_norm",      grad_norm, global_step)
                        writer.add_scalar("train/tokens_per_sec", tokens_s,  global_step)

                    if cfg.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "train/loss":           raw_loss,
                            "train/ppl":            ppl,
                            "train/acc":            acc,
                            "train/lr":             lr,
                            "train/grad_norm":      float(grad_norm),
                            "train/tokens_per_sec": tokens_s,
                        }, step=global_step)

                    t0 = time.perf_counter()

                # ── Evaluation ────────────────────────────────────────────────
                if global_step % cfg.eval_every == 0 and val_dl:
                    val_loss, val_acc = evaluate(
                        model, val_dl, device, amp_dtype
                    )
                    val_ppl = math.exp(min(val_loss, 88.0))  # ✅ FIX #13

                    logger.info(
                        "  [VAL] loss=%.4f  ppl=%.1f  acc=%.3f",
                        val_loss, val_ppl, val_acc,
                    )

                    if writer:
                        writer.add_scalar("val/loss", val_loss, global_step)
                        writer.add_scalar("val/ppl",  val_ppl,  global_step)
                        writer.add_scalar("val/acc",  val_acc,  global_step)

                    if cfg.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "val/loss": val_loss,
                            "val/ppl":  val_ppl,
                            "val/acc":  val_acc,
                        }, step=global_step)

                    # ── Best model ────────────────────────────────────────────
                    if val_loss < best_loss:
                        best_loss  = val_loss
                        no_improve = 0
                        best_path  = os.path.join(
                            cfg.model_save_dir, "best_model.pt"
                        )
                        
                        raw_model = getattr(model, "_orig_mod", model)
                        raw_model.save(best_path)
                        logger.info(
                            "  New best loss=%.4f -> %s",
                            best_loss, best_path,
                        )
                    else:
                        no_improve += 1
                        logger.info(
                            "  No improvement %d/%d",
                            no_improve, cfg.early_stop_patience,
                        )

                    
                    if no_improve >= cfg.early_stop_patience:
                        logger.info(
                            "Early stopping triggered at step=%d",
                            global_step,
                        )
                        stop_training = True
                        break

                    model.train()

                
                if not val_dl:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_path = os.path.join(
                            cfg.model_save_dir, "best_model.pt"
                        )
                        raw_model = getattr(model, "_orig_mod", model)
                        raw_model.save(best_path)

                # ── Periodic checkpoint ───────────────────────────────────────
                if global_step % cfg.save_every == 0:
                    ckpt_path = os.path.join(
                        cfg.model_save_dir,
                        f"step_{global_step:07d}.pt",
                    )
                    
                    torch.save(
                        {
                            "model_state":     model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scaler_state":    scaler.state_dict(),
                            "step":            global_step,
                            "epoch":           epoch,
                            "best_loss":       best_loss,
                            "no_improve":      no_improve,
                            "config":          _safe_config_dict(model),
                        },
                        ckpt_path,
                    )
                    logger.info("Checkpoint saved -> %s", ckpt_path)

                    # ── Rotate old checkpoints ────────────────────────────────
                    all_ckpts = sorted(
                        Path(cfg.model_save_dir).glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]),
                    )
                    for old in all_ckpts[: -cfg.keep_checkpoints]:
                        old.unlink(missing_ok=True)

            # ── Micro-batch metrics ───────────────────────────────────────────
            epoch_loss += loss.item()
            epoch_acc  += acc
            n_batches  += 1

        # ── End of epoch ──────────────────────────────────────────────────────
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc  = epoch_acc  / max(n_batches, 1)
        logger.info(
            "Epoch %d/%d  avg_loss=%.4f  avg_acc=%.3f  best_loss=%.4f",
            epoch + 1, cfg.epochs, avg_loss, avg_acc, best_loss,
        )

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(cfg.model_save_dir, "final.pt")
    raw_model  = getattr(model, "_orig_mod", model)
    raw_model.save(final_path)
    logger.info("Final checkpoint -> %s", final_path)

    # ── Training summary ──────────────────────────────────────────────────────
    summary = {
        "total_steps":   global_step,
        "best_val_loss": best_loss,
        "best_val_ppl":  math.exp(min(best_loss, 88.0)),
        "total_params":  total_params,
        "vocab_size":    tok.vocab_size,
        "device":        str(device),
        "epochs_ran":    cfg.epochs,
        "early_stopped": stop_training,
    }
    summary_path = os.path.join(
        cfg.model_save_dir, "training_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary -> %s", summary_path)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if writer:
        writer.close()
    if cfg.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    logger.info(
        "\n%s\n  Training Complete\n"
        "  total_steps   = %d\n"
        "  best_val_loss = %.4f\n"
        "  best_val_ppl  = %.1f\n%s",
        "=" * 45,
        global_step,
        best_loss,
        math.exp(min(best_loss, 88.0)),
        "=" * 45,
    )


if __name__ == "__main__":
    main()
