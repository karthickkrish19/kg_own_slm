import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    from tokenizers import (
        Tokenizer,
        models,
        pre_tokenizers,
        trainers,
        processors,
    )
    from tokenizers.normalizers import (
        NFD, Lowercase, StripAccents, Sequence
    )
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    logger.error("tokenizers not installed. Run: pip install tokenizers")


# ─────────────────────────────────────────────────────────────────────────────
# Special token constants
# List ORDER determines IDs → do not reorder
# ─────────────────────────────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────
def _verify_special_ids(tokenizer: "Tokenizer") -> None:
    expected = {
        PAD_TOKEN: PAD_ID,
        BOS_TOKEN: BOS_ID,
        EOS_TOKEN: EOS_ID,
        UNK_TOKEN: UNK_ID,
    }
    for token, exp_id in expected.items():
        actual_id = tokenizer.token_to_id(token)
        if actual_id != exp_id:
            raise RuntimeError(
                f"Special token '{token}' has id={actual_id}, "
                f"expected {exp_id}.\n"
                f"SPECIAL_TOKENS must be ordered: [PAD, BOS, EOS, UNK]"
            )
        
    logger.info(
        "Special token IDs verified: PAD=%d  BOS=%d  EOS=%d  UNK=%d",
        PAD_ID, BOS_ID, EOS_ID, UNK_ID,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
def train_tokenizer(
    files:         List[str],
    vocab_size:    int  = 7000,
    save_path:     str  = "out/tokenizer.json",
    min_frequency: int  = 2,
    lowercase:     bool = False,
) -> "SLMTokenizer":
    if not TOKENIZERS_AVAILABLE:
        raise ImportError("pip install tokenizers")

    valid_files = [f for f in files if Path(f).exists()]
    if not valid_files:
        raise FileNotFoundError(
            f"No corpus files found: {files}\n"
            "Run: python split_data.py"
        )
    logger.info(
        "Training BPE tokenizer | files=%s | vocab_size=%d",
        valid_files, vocab_size,
    )

    tokenizer = Tokenizer(models.BPE(
        unk_token = UNK_TOKEN,
        fuse_unk  = True,
    ))

    if lowercase:
        tokenizer.normalizer = Sequence(
            [NFD(), Lowercase(), StripAccents()]
        )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size       = vocab_size,
        min_frequency    = min_frequency,
        special_tokens   = SPECIAL_TOKENS,
        show_progress    = True,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(files=valid_files, trainer=trainer)

    _verify_special_ids(tokenizer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single        = f"{BOS_TOKEN}:0 $A:0 {EOS_TOKEN}:0",
        pair          = (
            f"{BOS_TOKEN}:0 $A:0 {EOS_TOKEN}:0 "
            f"$B:0 {EOS_TOKEN}:0"
        ),
        special_tokens = [
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    tokenizer.save(save_path)

    logger.info(
        "Tokenizer saved -> %s | vocab=%d | PAD=%d BOS=%d EOS=%d UNK=%d",
        save_path,
        tokenizer.get_vocab_size(),
        tokenizer.token_to_id(PAD_TOKEN),
        tokenizer.token_to_id(BOS_TOKEN),
        tokenizer.token_to_id(EOS_TOKEN),
        tokenizer.token_to_id(UNK_TOKEN),
    )

    return SLMTokenizer(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# SLMTokenizer
# ─────────────────────────────────────────────────────────────────────────────
class SLMTokenizer:
    # ─────────────────────────────────────────────────────────────────────────
    # Init
    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self, path: str) -> None:
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("pip install tokenizers")

        path = str(path)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {path}\n"
                "Train first:  python train.py"
            )

        
        self._tok  = Tokenizer.from_file(path)
        self._path = path

        self._tok.no_padding()
        self._tok.no_truncation()

        # Resolve IDs from loaded vocab
        self.pad_id = self._resolve_id(PAD_TOKEN, PAD_ID)
        self.bos_id = self._resolve_id(BOS_TOKEN, BOS_ID)
        self.eos_id = self._resolve_id(EOS_TOKEN, EOS_ID)
        self.unk_id = self._resolve_id(UNK_TOKEN, UNK_ID)

        # O(1) membership check for special tokens
        self._special_ids = frozenset(
            {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        )

        logger.debug(
            "SLMTokenizer loaded | vocab=%d | "
            "PAD=%d BOS=%d EOS=%d UNK=%d",
            self.vocab_size,
            self.pad_id, self.bos_id,
            self.eos_id, self.unk_id,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers — CLASS level, NOT inside __init__
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_id(self, token: str, fallback: int) -> int:

        result = self._tok.token_to_id(token)
        if result is None:
            logger.warning(
                "Special token '%s' missing from vocab — fallback id=%d",
                token, fallback,
            )
            return fallback
        return result

    def _safe_truncate(
        self,
        ids:                List[int],
        max_len:            int,
        add_special_tokens: bool,
    ) -> List[int]:
        
        if len(ids) <= max_len:
            return ids

        if add_special_tokens and max_len >= 3:
            keep = max_len - 2
            return [ids[0]] + ids[1 : 1 + keep] + [ids[-1]]

        return ids[:max_len]

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size() 

    @property
    def vocab(self) -> Dict[str, int]:
        return self._tok.get_vocab() 

    # ─────────────────────────────────────────────────────────────────────────
    # Encode
    # ─────────────────────────────────────────────────────────────────────────

    def encode(
        self,
        text:               str,
        add_special_tokens: bool          = True,
        max_len:            Optional[int] = None,
        truncate:           bool          = True,
    ) -> List[int]:
        enc = self._tok.encode(  
            text,
            add_special_tokens=add_special_tokens,
        )
        ids = enc.ids

        if max_len is not None and truncate:
            ids = self._safe_truncate(ids, max_len, add_special_tokens)

        return ids

    # ─────────────────────────────────────────────────────────────────────────
    # Decode
    # ─────────────────────────────────────────────────────────────────────────

    def decode(
        self,
        ids:          List[int],
        skip_special: bool = True,
    ) -> str:
        
        if not ids:
            return ""

        if skip_special:
            ids = [i for i in ids if i not in self._special_ids]

        if not ids:
            return ""
        
        return self._tok.decode(ids, skip_special_tokens=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Batch encode
    # ─────────────────────────────────────────────────────────────────────────

    def batch_encode(
        self,
        texts:              List[str],
        max_len:            int  = 256,
        pad:                bool = True,
        truncate:           bool = True,
        add_special_tokens: bool = True,
    ) -> Dict[str, List[List[int]]]:
        
        encodings  = self._tok.encode_batch(
            texts,
            add_special_tokens=add_special_tokens,
        )

        all_ids   = []
        all_masks = []

        for enc in encodings:
            ids = enc.ids

            if truncate:
                ids = self._safe_truncate(ids, max_len, add_special_tokens)

            real_len = len(ids)
            mask     = [1] * real_len

            if pad and real_len < max_len:
                gap  = max_len - real_len
                ids  = ids  + [self.pad_id] * gap
                mask = mask + [0]           * gap

            all_ids.append(ids)
            all_masks.append(mask)

        return {
            "input_ids":      all_ids,
            "attention_mask": all_masks,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # __call__ — HuggingFace-compatible interface
    # ─────────────────────────────────────────────────────────────────────────

    def __call__(
        self,
        texts:              Union[str, List[str]],
        max_len:            int  = 256,
        pad:                bool = True,
        truncate:           bool = True,
        add_special_tokens: bool = True,
    ) -> Dict[str, List[List[int]]]:
        
        if isinstance(texts, str):
            texts = [texts]

        return self.batch_encode(
            texts,
            max_len            = max_len,
            pad                = pad,
            truncate           = truncate,
            add_special_tokens = add_special_tokens,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Token utilities
    # ─────────────────────────────────────────────────────────────────────────

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)  

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self._tok.id_to_token(token_id) 

    def get_special_tokens_mask(self, ids: List[int]) -> List[int]:
        """1 for special tokens, 0 for regular tokens."""
        return [1 if i in self._special_ids else 0 for i in ids]

    def is_special_token(self, token_id: int) -> bool:
        return token_id in self._special_ids

    # ─────────────────────────────────────────────────────────────────────────
    # Dunder
    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        
        return (
            f"SLMTokenizer(\n"
            f"  path       = {self._path}\n"
            f"  vocab_size = {self.vocab_size}\n"
            f"  PAD={self.pad_id}  BOS={self.bos_id}  "
            f"EOS={self.eos_id}  UNK={self.unk_id}\n"
            f")"
        )
