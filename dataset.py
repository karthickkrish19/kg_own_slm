"""
SLM Dataset — Production / Enterprise Grade

Features:
  ✅ TokenizedDataset   — sliding window, correct off-by-one
  ✅ disk cache         — tokenize once, reuse across all runs
  ✅ stride support     — overlapping windows = more training data
  ✅ collate_fn         — pads + loss_mask (PAD excluded from loss)
  ✅ get_dataloaders()  — train / val / test loaders
  ✅ pin_memory         — CUDA only
  ✅ Windows safe       — num_workers=0 default
  ✅ Reproducible       — seeded generator
  ✅ No data leakage    — val/test stride = seq_len (no overlap)
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────
def slm_collate_fn(
    batch:  List[Tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate (x, y) pairs into a padded batch with loss mask.

    Why loss_mask matters:
        Without it, PAD tokens contribute to cross-entropy loss,
        making the model learn to predict PAD — not real language.

    Args:
        batch  : list of (x, y) from TokenizedDataset.__getitem__
        pad_id : token id used for padding  (must match tokenizer PAD_ID)

    Returns:
        x         : LongTensor  [B, T]   input ids
        y         : LongTensor  [B, T]   target ids
        loss_mask : BoolTensor  [B, T]   True=compute loss  False=skip
    """
    xs, ys  = zip(*batch)
    max_len = max(x.size(0) for x in xs)

    padded_x = []
    padded_y = []
    masks    = []

    for x, y in zip(xs, ys):
        pad_len = max_len - x.size(0)

        padded_x.append(
            torch.cat([
                x,
                torch.full((pad_len,), pad_id, dtype=torch.long),
            ])
        )
        padded_y.append(
            torch.cat([
                y,
                torch.full((pad_len,), pad_id, dtype=torch.long),
            ])
        )
        masks.append(
            torch.cat([
                torch.ones (x.size(0), dtype=torch.bool),
                torch.zeros(pad_len,   dtype=torch.bool),
            ])
        )

    return (
        torch.stack(padded_x),  # [B, T]
        torch.stack(padded_y),  # [B, T]
        torch.stack(masks),     # [B, T]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────
def _file_hash(file_path: str) -> str:
    """
    Stable MD5 hash of file — detects corpus changes.
    Cache is auto-invalidated if the file is modified.
    """
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _cache_path(
    file_path:  str,
    cache_dir:  str,
    vocab_size: int,
    tokenizer_path: str = "",
) -> Path:
    """
    Deterministic cache filename based on:
        - corpus file content hash
        - vocabulary size
        - tokenizer identity (path hash)

    If any of these change -> different cache file -> retokenized.
    """
    fhash = _file_hash(file_path)
    stem  = Path(file_path).stem
    # Include tokenizer path hash so retraining tokenizer invalidates cache
    tok_hash = ""
    if tokenizer_path and Path(tokenizer_path).exists():
        tok_hash = "_t" + _file_hash(tokenizer_path)
    return Path(cache_dir) / f"{stem}_{fhash}_v{vocab_size}{tok_hash}.pt"


# ─────────────────────────────────────────────────────────────────────────────
# TokenizedDataset
# ─────────────────────────────────────────────────────────────────────────────
class TokenizedDataset(Dataset):
    """
    Sliding-window next-token prediction dataset.

    How it works:
        1. Load corpus text file
        2. Tokenize entire file into one flat token tensor
        3. Create (x, y) windows:
               x = tokens[s   : s+seq_len]
               y = tokens[s+1 : s+seq_len+1]
           where y[i] = "what token follows x[i]"
        4. Stride controls overlap between windows:
               stride = seq_len      → no overlap (val/test)
               stride = seq_len // 2 → 50% overlap (2× more data)
               stride = seq_len // 4 → 75% overlap (4× more data)

    Off-by-one rule:
        Need tokens[s + seq_len] to exist for y.
        So: last valid s = len(tokens) - seq_len - 1
        And: num_chunks  = (len(tokens) - seq_len - 1) // stride + 1

    Args:
        file_path : path to plain-text corpus file
        tokenizer : SLMTokenizer instance
        seq_len   : context length in tokens
        stride    : step size between windows
        cache_dir : directory for cached token tensors
                    None = no caching (re-tokenize every run)
    """

    def __init__(
        self,
        file_path:  str,
        tokenizer,
        seq_len:    int           = 256,
        stride:     Optional[int] = None,
        cache_dir:  Optional[str] = "out/tokenized_cache",
    ) -> None:
        self.seq_len   = seq_len
        self.stride    = stride if stride is not None else seq_len
        self.file_path = str(file_path)

        # ── Validate file ─────────────────────────────────────────────────────
        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"Corpus file not found: {file_path}\n"
                "Run:  python split_data.py"
            )

        # ── Load tokens (from cache or fresh tokenization) ────────────────────
        self.tokens = self._load_tokens(
            Path(file_path), tokenizer, cache_dir
        )

        n = len(self.tokens)

        # ── Validate corpus size ──────────────────────────────────────────────
        # Need at least seq_len + 1 tokens so that y[-1] = tokens[seq_len]
        if n < seq_len + 1:
            raise ValueError(
                f"{file_path}: only {n} tokens available.\n"
                f"Need at least {seq_len + 1} tokens "
                f"(seq_len={seq_len} + 1 for targets).\n"
                f"Use a larger corpus or reduce block_size in config.yaml."
            )

        # ── Build window start indices ────────────────────────────────────────
        # last valid start = n - seq_len - 1  (so tokens[s+seq_len] exists)
        # range is O(1) memory — NOT a list
        self._starts = range(0, n - seq_len, self.stride)

        logger.info(
            "TokenizedDataset | %-20s | tokens=%s | "
            "seq_len=%d | stride=%d | chunks=%d",
            Path(file_path).name,
            f"{n:,}",
            seq_len,
            self.stride,
            len(self._starts),
        )

    # ── Token loading ─────────────────────────────────────────────────────────
    def _load_tokens(
        self,
        path:      Path,
        tokenizer,
        cache_dir: Optional[str],
    ) -> torch.Tensor:
        """
        Load token tensor from disk cache or tokenize fresh.

        Cache key = (file hash, vocab size, tokenizer identity)
        -> auto-invalidated if corpus, vocab, or tokenizer changes.
        """
        # ── Check cache ───────────────────────────────────────────────────────
        if cache_dir is not None:
            vocab_size = getattr(tokenizer, "vocab_size", 0)
            tok_path   = getattr(tokenizer, "_path", "")
            cached     = _cache_path(str(path), cache_dir, vocab_size, tok_path)

            if cached.exists():
                logger.info("Cache hit  -> %s", cached.name)
                tokens = torch.load(cached, weights_only=True)
                logger.info(
                    "Loaded %s tokens from cache", f"{len(tokens):,}"
                )
                return tokens

        # ── Tokenize fresh ────────────────────────────────────────────────────
        logger.info("Tokenizing %s ...", path.name)
        text = path.read_text(encoding="utf-8", errors="replace")

        if not text.strip():
            raise ValueError(
                f"{path} is empty. "
                "Add training data before running train.py"
            )

        # ✅ Call .encode() explicitly — NOT __call__
        #    __call__ returns dict {"input_ids":..., "attention_mask":...}
        #    .encode() returns List[int]
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,  # raw tokens only — no BOS/EOS
            truncate=False,            # never truncate corpus
        )

        if len(ids) == 0:
            raise ValueError(
                f"Tokenizer produced 0 tokens for {path}.\n"
                "Check tokenizer training and corpus encoding."
            )

        tokens = torch.tensor(ids, dtype=torch.long)

        # ── Save cache ────────────────────────────────────────────────────────
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(tokens, cached)
            logger.info(
                "Cache saved -> %s  (%s tokens)",
                cached.name, f"{len(tokens):,}",
            )

        return tokens

    # ── Dataset interface ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x : LongTensor [seq_len]   input tokens
            y : LongTensor [seq_len]   target tokens (x shifted right by 1)
        """
        s = self._starts[idx]
        x = self.tokens[s     : s + self.seq_len    ]
        y = self.tokens[s + 1 : s + self.seq_len + 1]
        return x, y

    # ── Stats helper ──────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Union[int, float]]:
        """Return dataset statistics dict — useful for logging."""
        n = len(self.tokens)
        return {
            "file_path"   : self.file_path,
            "total_tokens": n,
            "seq_len"     : self.seq_len,
            "stride"      : self.stride,
            "num_chunks"  : len(self._starts),
            "coverage"    : round(
                len(self._starts) * self.stride / max(n, 1), 4
            ),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"TokenizedDataset(\n"
            f"  file        = {s['file_path']}\n"
            f"  tokens      = {s['total_tokens']:,}\n"
            f"  seq_len     = {s['seq_len']}\n"
            f"  stride      = {s['stride']}\n"
            f"  chunks      = {s['num_chunks']:,}\n"
            f"  coverage    = {s['coverage']:.2%}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────────────────────────────────────
def get_dataloaders(
    train_file:  str,
    tokenizer,
    seq_len:     int           = 256,
    batch_size:  int           = 32,
    num_workers: int           = 0,
    stride:      Optional[int] = None,
    val_file:    Optional[str] = None,
    test_file:   Optional[str] = None,
    seed:        int           = 42,
    cache_dir:   Optional[str] = "out/tokenized_cache",
    pad_id:      int           = 0,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Build train / val / test DataLoaders.

    Design decisions:
        train  → shuffle=True,  drop_last=True,  stride=stride (overlap OK)
        val    → shuffle=False, drop_last=False, stride=seq_len (no overlap)
        test   → shuffle=False, drop_last=False, stride=seq_len (no overlap)

    Args:
        train_file  : path to train corpus .txt
        tokenizer   : SLMTokenizer instance
        seq_len     : context window size in tokens
        batch_size  : sequences per batch
        num_workers : parallel data workers (0 = safe on Windows)
        stride      : train window stride
                      None → seq_len (no overlap)
                      seq_len//2 → 50% overlap (recommended)
        val_file    : path to val corpus .txt   (optional)
        test_file   : path to test corpus .txt  (optional)
        seed        : RNG seed for reproducibility
        cache_dir   : token cache directory
        pad_id      : padding token id

    Returns:
        (train_dl, val_dl, test_dl)
        val_dl and test_dl are None if not provided
    """
    # ── CUDA check for pin_memory ─────────────────────────────────────────────
    pin = torch.cuda.is_available()

    def collate(batch):
        return slm_collate_fn(batch, pad_id=pad_id)

    # ── Train DataLoader ──────────────────────────────────────────────────────
    train_ds = TokenizedDataset(
        file_path = train_file,
        tokenizer = tokenizer,
        seq_len   = seq_len,
        stride    = stride,         # overlap for more training data
        cache_dir = cache_dir,
    )

    train_dl = DataLoader(
        dataset            = train_ds,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = num_workers,
        pin_memory         = pin,
        drop_last          = True,   # avoids small last batch issues
        collate_fn         = collate,
        generator          = torch.Generator().manual_seed(seed),
        persistent_workers = num_workers > 0,
    )

    # ── Val DataLoader ────────────────────────────────────────────────────────
    val_dl = None
    if val_file and Path(val_file).exists():
        val_ds = TokenizedDataset(
            file_path = val_file,
            tokenizer = tokenizer,
            seq_len   = seq_len,
            stride    = seq_len,    # no overlap → no data leakage
            cache_dir = cache_dir,
        )
        val_dl = DataLoader(
            dataset            = val_ds,
            batch_size         = batch_size,
            shuffle            = False,
            num_workers        = num_workers,
            pin_memory         = pin,
            drop_last          = False,
            collate_fn         = collate,
            persistent_workers = num_workers > 0,
        )
    else:
        logger.warning(
            "No val_file provided or not found: %s\n"
            "Validation metrics will be skipped during training.",
            val_file,
        )

    # ── Test DataLoader ───────────────────────────────────────────────────────
    test_dl = None
    if test_file and Path(test_file).exists():
        test_ds = TokenizedDataset(
            file_path = test_file,
            tokenizer = tokenizer,
            seq_len   = seq_len,
            stride    = seq_len,    # no overlap
            cache_dir = cache_dir,
        )
        test_dl = DataLoader(
            dataset            = test_ds,
            batch_size         = batch_size,
            shuffle            = False,
            num_workers        = num_workers,
            pin_memory         = pin,
            drop_last          = False,
            collate_fn         = collate,
            persistent_workers = num_workers > 0,
        )

    # ── Summary log ───────────────────────────────────────────────────────────
    def _nb(dl): return f"{len(dl):,}" if dl else "N/A"
    def _nc(ds): return f"{len(ds):,}" if ds else "N/A"

    logger.info(
        "\n%s\n  DataLoaders Ready\n%s\n"
        "  train  batches=%-8s chunks=%s\n"
        "  val    batches=%-8s\n"
        "  test   batches=%-8s\n"
        "  batch_size   = %d\n"
        "  num_workers  = %d\n"
        "  pin_memory   = %s\n"
        "  cache_dir    = %s\n%s",
        "=" * 45, "=" * 45,
        _nb(train_dl), _nc(train_ds),
        _nb(val_dl),
        _nb(test_dl),
        batch_size,
        num_workers,
        pin,
        cache_dir,
        "=" * 45,
    )

    return train_dl, val_dl, test_dl
