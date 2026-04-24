"""
SLM Inference Engine
====================
Production-ready inference with safety filtering, streaming, embeddings,
and perplexity scoring.

Usage:
    from inference import SLMInference

    engine = SLMInference(cfg, "out/checkpoints/best.pt")
    text = engine.generate("Hello world")
    for chunk in engine.stream("Tell me a story"):
        print(chunk, end="", flush=True)
"""

import math
import re
import logging
from pathlib import Path
from typing import List, Optional, Union

import torch

logger = logging.getLogger(__name__)


class SLMInference:
    """Full inference engine: generate, stream, embed, perplexity."""

    def __init__(
        self,
        config,
        checkpoint_path: str,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = config

        # ── Device ────────────────────────────────────────────────────────────
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # ── Tokenizer ─────────────────────────────────────────────────────────
        from tokenizer_utils import SLMTokenizer

        tok_path = config.tokenizer_file
        if not Path(tok_path).exists():
            raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
        self.tok = SLMTokenizer(tok_path)

        # ── Model ─────────────────────────────────────────────────────────────
        from model import SLM

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.model = SLM.load(checkpoint_path, map_location=str(self.device))
        self.model.to(self.device)
        self.model.eval()

        # ── Safety filter ─────────────────────────────────────────────────────
        self._safety_enabled = config.enable_safety_filter
        self._safety_patterns = [
            re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            for kw in config.safety_keywords
        ]

        logger.info(
            "SLMInference ready | device=%s | vocab=%d | safety=%s",
            self.device, self.tok.vocab_size, self._safety_enabled,
        )

    # ── Safety check ──────────────────────────────────────────────────────────
    def _is_safe(self, text: str) -> bool:
        if not self._safety_enabled:
            return True
        for pat in self._safety_patterns:
            if pat.search(text):
                return False
        return True

    # ── Encode prompt ─────────────────────────────────────────────────────────
    def _encode_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_len: int = 200,
    ) -> torch.Tensor:
        if context:
            full = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
        else:
            full = prompt
        # Model was trained WITHOUT special tokens (dataset uses add_special_tokens=False)
        # so inference must also NOT add BOS/EOS
        ids = self.tok.encode(
            full, add_special_tokens=False, max_len=max_len, truncate=True
        )
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    # ── Generate ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        repetition_window: Optional[int] = None,
        context: Optional[str] = None,
        skip_special_tokens: bool = True,
    ) -> str:
        if not self._is_safe(prompt):
            return "[Content filtered]"

        max_new = max_new_tokens or self.cfg.max_new_tokens
        temp = temperature if temperature is not None else self.cfg.temperature
        tk = top_k if top_k is not None else self.cfg.top_k
        tp = top_p if top_p is not None else self.cfg.top_p
        rp = repetition_penalty if repetition_penalty is not None else self.cfg.repetition_penalty
        rw = repetition_window if repetition_window is not None else self.cfg.repetition_window

        max_ctx = max(8, self.cfg.block_size - max_new)
        ids = self._encode_prompt(prompt, context, max_len=max_ctx)

        # Use EOS for natural stopping if config says so
        eos_id = self.tok.eos_id if getattr(self.cfg, 'stop_at_eos', False) else None

        out_ids = self.model.generate(
            idx=ids,
            max_new_tokens=max_new,
            temperature=temp,
            top_k=tk,
            top_p=tp,
            repetition_penalty=rp,
            repetition_window=rw,
            eos_id=eos_id,
        )

        new_tokens = out_ids[0, ids.shape[1]:].tolist()
        text = self.tok.decode(new_tokens, skip_special=skip_special_tokens)

        if not self._is_safe(text):
            return "[Content filtered]"
        return text

    # ── Stream generate ───────────────────────────────────────────────────────
    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        context: Optional[str] = None,
    ):
        if not self._is_safe(prompt):
            yield "[Content filtered]"
            return

        max_new = max_new_tokens or self.cfg.max_new_tokens
        temp = temperature if temperature is not None else self.cfg.temperature
        tk = top_k if top_k is not None else self.cfg.top_k
        tp = top_p if top_p is not None else self.cfg.top_p

        max_ctx = max(8, self.cfg.block_size - max_new)
        ids = self._encode_prompt(prompt, context, max_len=max_ctx)

        eos_id = self.tok.eos_id if getattr(self.cfg, 'stop_at_eos', False) else None

        for token_id in self.model.stream_generate(
            idx=ids,
            max_new_tokens=max_new,
            temperature=temp,
            top_k=tk,
            top_p=tp,
            eos_id=eos_id,
        ):
            token_text = self.tok.decode([token_id], skip_special=True)
            if token_text:
                yield token_text

    # ── Embeddings ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def embed(
        self,
        texts: Union[str, List[str]],
        pooling: str = "mean",
        batch_size: int = 32,
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        max_len = self.cfg.block_size
        all_embeds = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tok.batch_encode(
                batch, max_len=max_len, pad=True, truncate=True,
                add_special_tokens=False,
            )
            ids_list = encoded["input_ids"]
            ids_tensor = torch.tensor(
                ids_list, dtype=torch.long, device=self.device
            )
            emb = self.model.get_embeddings(ids_tensor, pooling=pooling)
            all_embeds.append(emb.cpu())

        return torch.cat(all_embeds, dim=0)

    # ── Perplexity ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            return float("inf")
        max_len = self.cfg.block_size
        ids = ids[:max_len]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=self.device)
        _, loss, _ = self.model(x, targets=y)
        return math.exp(min(loss.item(), 88.0))
