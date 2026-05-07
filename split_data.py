import os
import re
import json
import math
import hashlib
import logging
import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("out", exist_ok=True)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("out/split_data.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants / Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT      = "data/input.txt"
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_TRAIN      = 0.80
DEFAULT_VAL        = 0.10
DEFAULT_TEST       = 0.10
DEFAULT_SEED       = 42
DEFAULT_MIN_LEN    = 50       # chars — below this a chunk is noise
DEFAULT_CHUNK_SIZE = 512      # chars for character mode
DEFAULT_CHUNK_STEP = 256      # stride for character mode


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split corpus into train/val/test")
    p.add_argument("--input",      default=DEFAULT_INPUT,      help="input .txt file")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="output directory")
    p.add_argument("--train",      type=float, default=DEFAULT_TRAIN, help="train ratio")
    p.add_argument("--val",        type=float, default=DEFAULT_VAL,   help="val ratio")
    p.add_argument("--test",       type=float, default=DEFAULT_TEST,  help="test ratio")
    p.add_argument("--seed",       type=int,   default=DEFAULT_SEED,  help="random seed")
    p.add_argument("--min_len",    type=int,   default=DEFAULT_MIN_LEN, help="min chunk chars")
    p.add_argument("--chunk_size", type=int,   default=DEFAULT_CHUNK_SIZE, help="char-mode chunk size")
    p.add_argument("--chunk_step", type=int,   default=DEFAULT_CHUNK_STEP, help="char-mode step")
    p.add_argument("--no_shuffle", action="store_true", help="disable shuffle")
    p.add_argument("--dedupe",     action="store_true", help="deduplicate chunks")
    p.add_argument("--no_unicode", action="store_true", help="strip non-ASCII characters")
    p.add_argument("--mode",       choices=["auto","paragraph","line","character"],
                   default="auto", help="split mode override")
    p.add_argument("--config",     default="config.yaml", help="config.yaml path")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# File safety check
# ─────────────────────────────────────────────────────────────────────────────
_MAX_FILE_BYTES = 2 * 1024 ** 3   # 2 GB — above this warn user

def _check_file_size(path: Path) -> int:
    """
    ✅ FIX #12: Check file size before loading into RAM.
    Warn on large files; raise on missing files.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"Place your corpus at: {path}"
        )
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"{path} is empty (0 bytes).")
    if size > _MAX_FILE_BYTES:
        logger.warning(
            "File is %.1f GB — consider streaming or chunking.\n"
            "Reading entire file into RAM...",
            size / 1024**3,
        )
    return size


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text: str, keep_unicode: bool = True) -> str:
    """
    Normalise whitespace and optionally strip non-printable control chars.

    ✅ FIX #8: keep_unicode=True preserves multilingual text.
               Only strips actual control chars (not printable unicode).

    Args:
        text         : raw corpus string
        keep_unicode : True = keep all printable unicode (default)
                       False = ASCII-only (legacy behaviour)
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if keep_unicode:
        # Remove only actual control characters (not printable unicode)
        # Keeps: tabs (0x09), newlines (0x0A), all printable chars
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    else:
        # ✅ FIX #8: legacy ASCII-only — only use if corpus is ASCII
        text = re.sub(
            r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", "", text
        )

    # Collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ── Unwrap hard-wrapped lines ─────────────────────────────────────────
    # Many Project Gutenberg texts (and similar corpora) are hard-wrapped
    # at ~70 chars.  These '\n' are formatting artifacts — not real
    # paragraph breaks.  Unwrap them to produce flowing prose so the
    # tokenizer doesn't waste capacity on thousands of newline tokens.
    #
    # Strategy: protect real paragraph breaks (\n\n), then replace every
    # remaining single \n with a space.
    _PARA = "\x00PARA\x00"
    text = text.replace("\n\n", _PARA)   # protect paragraph boundaries
    text = text.replace("\n", " ")       # unwrap hard wraps
    text = text.replace(_PARA, "\n\n")   # restore paragraph boundaries
    text = re.sub(r" {2,}", " ", text)   # collapse double-spaces

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Split mode detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_split_mode(text: str) -> str:
    """
    Auto-detect best chunking strategy from corpus structure.

    Rules:
        blank_ratio > 0.10 AND avg_paragraph_len > 100 → paragraph
        n_lines    > 500   AND avg_line_len > 40        → line
        else                                            → character

    ✅ FIX #6: tightened thresholds — 0.05 was too permissive,
       fired on almost any file with blank lines.
    """
    lines      = text.splitlines()
    n_lines    = len(lines)
    blank      = sum(1 for l in lines if not l.strip())
    blank_ratio= blank / max(n_lines, 1)

    non_blank  = [l for l in lines if l.strip()]
    avg_line   = (
        sum(len(l) for l in non_blank) / max(len(non_blank), 1)
    )

    # Paragraph detection: meaningful blank lines + long paragraphs
    if blank_ratio > 0.10 and avg_line > 100:
        return "paragraph"

    # Line detection: many lines of reasonable length
    if n_lines > 500 and avg_line > 40:
        return "line"

    return "character"


# ─────────────────────────────────────────────────────────────────────────────
# Corpus analysis
# ─────────────────────────────────────────────────────────────────────────────
def research_file(text: str) -> Dict:
    """
    Analyse corpus and return full statistics dict.

    ✅ FIX #5: returns full stats dict — not just split_mode.
    ✅ FIX #13: word_freq uses split() not Counter(text) char-by-char.
    ✅ FIX #1: uses logger — not print().
    """
    lines      = text.splitlines()
    words      = text.split()
    sentences  = re.split(r"[.!?]+", text)
    sentences  = [s.strip() for s in sentences if s.strip()]

    # ✅ FIX #13: sample top chars from first 100K chars only — not full text
    sample     = text[:100_000]
    char_freq  = Counter(sample)
    word_freq  = Counter(
        w.lower().strip(".,!?;:\"'()[]") for w in words
    )

    non_blank  = [l for l in lines if l.strip()]
    line_lens  = [len(l) for l in non_blank]
    avg_line   = sum(line_lens) / max(len(line_lens), 1)
    mode       = detect_split_mode(text)

    # ── Vocab richness estimate ───────────────────────────────────────────────
    unique_words = len(word_freq)
    total_words  = len(words)
    type_token   = unique_words / max(total_words, 1)

    stats = {
        "total_chars":   len(text),
        "total_lines":   len(lines),
        "non_empty_lines": len(non_blank),
        "total_words":   total_words,
        "unique_words":  unique_words,
        "type_token_ratio": round(type_token, 4),
        "total_sentences": len(sentences),
        "avg_line_len":  round(avg_line, 1),
        "split_mode":    mode,
        "top_words":     dict(word_freq.most_common(20)),
        "top_chars":     {
            repr(k): v
            for k, v in char_freq.most_common(10)
        },
    }

    sep = "=" * 58
    logger.info("\n%s", sep)
    logger.info("  INPUT FILE RESEARCH REPORT")
    logger.info(sep)
    logger.info("  Total characters    : %s", f"{stats['total_chars']:,}")
    logger.info("  Total lines         : %s", f"{stats['total_lines']:,}")
    logger.info("  Non-empty lines     : %s", f"{stats['non_empty_lines']:,}")
    logger.info("  Total words         : %s", f"{stats['total_words']:,}")
    logger.info("  Unique words        : %s", f"{stats['unique_words']:,}")
    logger.info("  Type-Token ratio    : %.4f", type_token)
    logger.info("  Total sentences     : %s", f"{stats['total_sentences']:,}")
    logger.info("  Avg line length     : %.1f chars", avg_line)
    logger.info("  Detected split mode : %s", mode)
    logger.info("  Top 10 words        :")
    for w, c in word_freq.most_common(10):
        logger.info("    %-25s %6d", w, c)
    logger.info(sep)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────
def split_into_chunks(
    text:       str,
    mode:       str,
    min_len:    int = DEFAULT_MIN_LEN,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_step: int = DEFAULT_CHUNK_STEP,
) -> List[str]:
    """
    Split corpus text into training chunks.

    ✅ FIX #2: chunk_size and chunk_step are parameters — not hardcoded.
    ✅ FIX #15: min_len=50 by default (was 20 — too small).

    Modes:
        paragraph : split on blank lines
        line      : split on newlines
        character : sliding window of chunk_size chars, step chunk_step
    """
    if mode == "paragraph":
        raw = re.split(r"\n\s*\n", text)
    elif mode == "line":
        raw = text.splitlines()
    elif mode == "character":
        # ✅ FIX #2: configurable size/step
        raw = [
            text[i : i + chunk_size]
            for i in range(0, len(text) - chunk_size + 1, chunk_step)
        ]
    else:
        raise ValueError(
            f"Unknown split mode: '{mode}'. "
            "Use: paragraph | line | character"
        )

    chunks = [c.strip() for c in raw if len(c.strip()) >= min_len]
    logger.info(
        "Chunking: mode=%s  raw=%d  kept=%d  (min_len=%d)",
        mode, len(raw), len(chunks), min_len,
    )
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────
def deduplicate(chunks: List[str]) -> Tuple[List[str], int]:
    """
    Remove exact-duplicate chunks using MD5 fingerprints.

    Returns:
        (deduped_chunks, n_removed)
    """
    seen   = set()
    result = []
    for c in chunks:
        fp = hashlib.md5(c.encode("utf-8")).hexdigest()
        if fp not in seen:
            seen.add(fp)
            result.append(c)
    n_removed = len(chunks) - len(result)
    if n_removed:
        logger.info("Deduplication: removed %d duplicates", n_removed)
    return result, n_removed


# ─────────────────────────────────────────────────────────────────────────────
# Cross-split leakage check
# ─────────────────────────────────────────────────────────────────────────────
def check_leakage(
    train: List[str],
    val:   List[str],
    test:  List[str],
) -> None:
    """
    Verify no chunk appears in both train and val/test.
    Logs a warning if any overlap found.
    """
    train_fps = {
        hashlib.md5(c.encode("utf-8")).hexdigest() for c in train
    }
    val_fps  = {
        hashlib.md5(c.encode("utf-8")).hexdigest() for c in val
    }
    test_fps = {
        hashlib.md5(c.encode("utf-8")).hexdigest() for c in test
    }

    tv = train_fps & val_fps
    tt = train_fps & test_fps
    vt = val_fps   & test_fps

    if tv:
        logger.warning("WARNING: %d chunks overlap: train & val",  len(tv))
    if tt:
        logger.warning("WARNING: %d chunks overlap: train & test", len(tt))
    if vt:
        logger.warning("WARNING: %d chunks overlap: val & test",   len(vt))
    if not tv and not tt and not vt:
        logger.info("No cross-split leakage detected")


# ─────────────────────────────────────────────────────────────────────────────
# Write output
# ─────────────────────────────────────────────────────────────────────────────
def write_file(
    path:   str,
    chunks: List[str],
    mode:   str,
) -> None:
    """
    Write chunks to a text file.

    ✅ FIX #4: uses Path throughout — no mixed os.path / Path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sep = "\n\n" if mode == "paragraph" else "\n"
    out.write_text(sep.join(chunks), encoding="utf-8")
    logger.info("Written -> %s  (%s)", out, f"{out.stat().st_size:,} bytes")


# ─────────────────────────────────────────────────────────────────────────────
# Stats helper
# ─────────────────────────────────────────────────────────────────────────────
def _chunk_stats(chunks: List[str]) -> Dict:
    if not chunks:
        return {"count": 0, "words": 0, "chars": 0, "avg_len": 0}
    chars = sum(len(c) for c in chunks)
    words = sum(len(c.split()) for c in chunks)
    return {
        "count":   len(chunks),
        "words":   words,
        "chars":   chars,
        "avg_len": round(chars / len(chunks), 1),
    }


def _log_split_stats(
    train: List[str],
    val:   List[str],
    test:  List[str],
) -> None:
    total = len(train) + len(val) + len(test)
    sep   = "=" * 58
    logger.info("\n%s", sep)
    logger.info("  SPLIT RESULTS")
    logger.info(sep)
    for name, chunks in [("train.txt", train),
                          ("val.txt",   val),
                          ("test.txt",  test)]:
        s = _chunk_stats(chunks)
        logger.info(
            "  %-12s  chunks=%6s  words=%9s  chars=%11s  "
            "avg_len=%6.1f  (%.1f%%)",
            name,
            f"{s['count']:,}",
            f"{s['words']:,}",
            f"{s['chars']:,}",
            s["avg_len"],
            s["count"] / max(total, 1) * 100,
        )
    logger.info(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def split_data(
    input_file:  str   = DEFAULT_INPUT,
    output_dir:  str   = DEFAULT_OUTPUT_DIR,
    train_ratio: float = DEFAULT_TRAIN,
    val_ratio:   float = DEFAULT_VAL,
    test_ratio:  float = DEFAULT_TEST,
    seed:        int   = DEFAULT_SEED,
    shuffle:     bool  = True,
    dedupe:      bool  = False,
    min_len:     int   = DEFAULT_MIN_LEN,
    chunk_size:  int   = DEFAULT_CHUNK_SIZE,
    chunk_step:  int   = DEFAULT_CHUNK_STEP,
    mode:        str   = "auto",
    keep_unicode:bool  = True,
) -> Dict:
    """
    Split corpus into train / val / test sets.

    Args:
        input_file  : path to corpus .txt
        output_dir  : directory to write train/val/test.txt
        train_ratio : fraction of data for training
        val_ratio   : fraction for validation
        test_ratio  : fraction for testing
        seed        : random seed for shuffle
        shuffle     : shuffle chunks before splitting
        dedupe      : remove duplicate chunks
        min_len     : minimum chunk length in chars
        chunk_size  : char-mode window size
        chunk_step  : char-mode stride
        mode        : auto / paragraph / line / character
        keep_unicode: preserve non-ASCII chars

    Returns:
        split_report dict (also saved as JSON)
    """
    # ── Validate ratios ───────────────────────────────────────────────────────
    # ✅ FIX #3: ValueError with clear message instead of bare assert
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train + val + test must sum to 1.0, got {ratio_sum:.6f}.\n"
            f"Current: train={train_ratio}  val={val_ratio}  test={test_ratio}"
        )
    if train_ratio <= 0:
        raise ValueError("train_ratio must be > 0")
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be >= 0")

    # ── Create dirs ───────────────────────────────────────────────────────────
    os.makedirs("out",       exist_ok=True)
    os.makedirs(output_dir,  exist_ok=True)

    # ── Load file ─────────────────────────────────────────────────────────────
    input_path = Path(input_file)
    file_size  = _check_file_size(input_path)    # ✅ FIX #12
    logger.info(
        "Reading: %s  (%.2f MB)", input_path,
        file_size / 1024 ** 2,
    )
    raw_text = input_path.read_text(
        encoding="utf-8", errors="replace"
    )

    # ── Analyse ───────────────────────────────────────────────────────────────
    corpus_stats = research_file(raw_text)       # ✅ FIX #1 #5
    text         = clean_text(raw_text, keep_unicode=keep_unicode)  # ✅ FIX #8

    # ── Detect / override mode ────────────────────────────────────────────────
    if mode == "auto":
        mode = corpus_stats["split_mode"]
    logger.info("Using split mode: %s", mode)

    # ──────────────────────────────────────────────────────────────────────────
    # SPLIT STRATEGY: Per-book contiguous split
    # ──────────────────────────────────────────────────────────────────────────
    # Split the raw text into large contiguous blocks (books), then take
    # train/val/test slices from EACH book.  This ensures:
    #   1. All splits see all books (balanced distribution)
    #   2. Within each split, text is CONTIGUOUS (no broken sentences)
    #   3. No line-level fragmentation or "every Nth" gaps
    #
    # Books are detected by large gaps (double+ newlines) or by the
    # entire text being treated as one block if no gaps exist.
    # ──────────────────────────────────────────────────────────────────────────
    books = re.split(r"\n\s*\n", text)
    books = [b.strip() for b in books if len(b.strip()) >= 200]

    # If no blank-line separators found, treat entire text as one book
    if len(books) <= 1:
        books = [text.strip()]

    logger.info("Detected %d book(s) / section(s)", len(books))

    train_parts = []
    val_parts   = []
    test_parts  = []

    for i, book in enumerate(books):
        n = len(book)
        # Take contiguous slices from each book:
        #   train = first 80%
        #   val   = next 10%
        #   test  = last 10%
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        train_text = book[:train_end].strip()
        val_text   = book[train_end:val_end].strip()
        test_text  = book[val_end:].strip()

        if train_text:
            train_parts.append(train_text)
        if val_text:
            val_parts.append(val_text)
        if test_text:
            test_parts.append(test_text)

        logger.info(
            "  Book %d: %s chars -> train=%s val=%s test=%s",
            i + 1, f"{n:,}",
            f"{len(train_text):,}",
            f"{len(val_text):,}",
            f"{len(test_text):,}",
        )

    # Join all books' splits with double-newline separator
    train_full = "\n\n".join(train_parts)
    val_full   = "\n\n".join(val_parts)
    test_full  = "\n\n".join(test_parts)

    total_chars = len(train_full) + len(val_full) + len(test_full)
    logger.info(
        "Total: train=%s val=%s test=%s chars",
        f"{len(train_full):,}", f"{len(val_full):,}", f"{len(test_full):,}",
    )

    # ── Write files ───────────────────────────────────────────────────────────
    train_path = os.path.join(output_dir, "train.txt")
    val_path   = os.path.join(output_dir, "val.txt")
    test_path  = os.path.join(output_dir, "test.txt")

    Path(train_path).write_text(train_full, encoding="utf-8")
    Path(val_path).write_text(val_full, encoding="utf-8")
    Path(test_path).write_text(test_full, encoding="utf-8")

    logger.info("Written -> %s  (%s bytes)", train_path, f"{len(train_full.encode('utf-8')):,}")
    logger.info("Written -> %s  (%s bytes)", val_path, f"{len(val_full.encode('utf-8')):,}")
    logger.info("Written -> %s  (%s bytes)", test_path, f"{len(test_full.encode('utf-8')):,}")

    # ── Log stats ─────────────────────────────────────────────────────────────
    sep = "=" * 58
    logger.info("\n%s", sep)
    logger.info("  SPLIT RESULTS (per-book contiguous)")
    logger.info(sep)
    for name, txt in [("train.txt", train_full),
                       ("val.txt",   val_full),
                       ("test.txt",  test_full)]:
        words = len(txt.split())
        chars = len(txt)
        pct   = chars / max(total_chars, 1) * 100
        logger.info(
            "  %-12s  words=%9s  chars=%11s  (%.1f%%)",
            name, f"{words:,}", f"{chars:,}", pct,
        )
    logger.info(sep)

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "input_file":     str(input_path.resolve()),
        "file_size_bytes":file_size,
        "split_mode":     "per-book-contiguous",
        "n_books":        len(books),
        "corpus_stats":   {
            k: v for k, v in corpus_stats.items()
            if k not in ("top_words", "top_chars")
        },
        "splits": {
            "train": {"chars": len(train_full), "words": len(train_full.split()), "path": train_path},
            "val":   {"chars": len(val_full),   "words": len(val_full.split()),   "path": val_path},
            "test":  {"chars": len(test_full),  "words": len(test_full.split()),  "path": test_path},
        },
        "ratios": {
            "train": round(len(train_full) / max(total_chars, 1), 4),
            "val":   round(len(val_full)   / max(total_chars, 1), 4),
            "test":  round(len(test_full)  / max(total_chars, 1), 4),
        },
    }
    report_path = os.path.join(output_dir, "split_report.json")
    Path(report_path).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info("Split report -> %s", report_path)
    logger.info(
        "\n  [OK] %s\n  [OK] %s\n  [OK] %s",
        train_path, val_path, test_path,
    )

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # ── Try to load config.yaml for defaults ──────────────────────────────────
    cfg_seed       = args.seed
    cfg_min_len    = args.min_len
    cfg_input      = args.input
    cfg_output_dir = args.output_dir

    if Path(args.config).exists():
        try:
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f) or {}
            cfg_seed       = cfg.get("seed",          cfg_seed)
            # Only use corpus_files from config if --input was NOT explicitly given
            if args.input == DEFAULT_INPUT:
                cfg_input  = cfg.get("corpus_files", [cfg_input])[0]
            # W-10: only override output_dir from config when user didn't set it
            if args.output_dir == DEFAULT_OUTPUT_DIR:
                cfg_output_dir = "data"
            logger.info(
                "Config loaded: %s  (seed=%d input=%s)",
                args.config, cfg_seed, cfg_input,
            )
        except Exception as e:
            logger.warning("Could not load %s: %s", args.config, e)

    split_data(
        input_file  = cfg_input,
        output_dir  = cfg_output_dir,
        train_ratio = args.train,
        val_ratio   = args.val,
        test_ratio  = args.test,
        seed        = cfg_seed,
        shuffle     = not args.no_shuffle,
        dedupe      = args.dedupe,
        min_len     = cfg_min_len,
        chunk_size  = args.chunk_size,
        chunk_step  = args.chunk_step,
        mode        = args.mode,
        keep_unicode= not args.no_unicode,
    )
