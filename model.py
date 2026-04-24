import os
import math
import logging
from dataclasses import dataclass, asdict, fields
from typing import Dict, Generator, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

# ── Optional backends ─────────────────────────────────────────────────────────
HAS_FLASH_ATTN = False
try:
    if torch.cuda.is_available():
        from flash_attn import flash_attn_func          # type: ignore
        HAS_FLASH_ATTN = True
except ImportError:
    pass

HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


# ─────────────────────────────────────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SLMConfig:
    # Vocabulary
    vocab_size:             int   = 7000
    pad_id:                 int   = 0

    # Architecture
    max_seq_len:            int   = 256
    embed_dim:              int   = 256
    num_layers:             int   = 6
    num_heads:              int   = 8
    num_kv_heads:           int   = 2      
    ffn_multiplier:         float = 2.6667
    multiple_of:            int   = 64
    norm_eps:               float = 1e-6

    # Regularisation
    dropout:                float = 0.0
    attn_dropout:           float = 0.0
    ffn_dropout:            float = 0.0

    # RoPE
    rope_base:              int   = 10_000
    cache_factor:           int   = 4

    # Inference
    use_kv_cache:           bool  = True
    use_flash:              bool  = False

    # Training
    tie_weights:            bool  = True
    gradient_checkpointing: bool  = False
    init_scale:             float = 0.02

    def __post_init__(self) -> None:
        assert self.embed_dim % self.num_heads    == 0, \
            f"embed_dim {self.embed_dim} must be divisible by num_heads {self.num_heads}"
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads {self.num_heads} must be divisible by num_kv_heads {self.num_kv_heads}"
        assert self.vocab_size  > 0
        assert self.num_layers  > 0
        assert self.max_seq_len > 0
        assert 0.0 <= self.dropout      < 1.0
        assert 0.0 <= self.attn_dropout < 1.0
        assert 0.0 <= self.ffn_dropout  < 1.0
        assert self.num_kv_heads >= 1, \
            "num_kv_heads=1 → MQA   num_kv_heads=num_heads → MHA"


# ─────────────────────────────────────────────────────────────────────────────
# 1. RMSNorm
# ─────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        return self._norm(x.float()).to(x.dtype) * self.weight


# ─────────────────────────────────────────────────────────────────────────────
# 2. RoPE — Rotary Position Embeddings
# ─────────────────────────────────────────────────────────────────────────────
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q:           torch.Tensor,
    k:           torch.Tensor,
    cos:         torch.Tensor,
    sin:         torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K tensors."""
    # [T, head_dim] → [1, 1, T, head_dim]
    cos = cos[position_ids].unsqueeze(0).unsqueeze(0)
    sin = sin[position_ids].unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        dim:          int,
        max_seq_len:  int = 2048,
        base:         int = 10_000,
        cache_factor: int = 4,
    ) -> None:
        super().__init__()
        self.dim          = dim
        self.base         = base
        self.cache_factor = cache_factor

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2, dtype=torch.float32) / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len * cache_factor)

    def _build_cache(self, seq_len: int) -> None:
        t     = torch.arange(
            seq_len,
            device=self.inv_freq.device,
            dtype=torch.float32,
        )
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len * self.cache_factor)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. KV Cache
# ─────────────────────────────────────────────────────────────────────────────
class KVCache:

    def __init__(
        self,
        batch_size:   int,
        num_kv_heads: int,
        max_seq_len:  int,
        head_dim:     int,
        device:       torch.device,
        dtype:        torch.dtype,
    ) -> None:
        shape        = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.cache_k = torch.zeros(shape, device=device, dtype=dtype)
        self.cache_v = torch.zeros(shape, device=device, dtype=dtype)
        self.length  = 0

    def update(
        self,
        k:         torch.Tensor,
        v:         torch.Tensor,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = k.size(2)
        self.cache_k[:, :, start_pos : start_pos + T] = k.detach()
        self.cache_v[:, :, start_pos : start_pos + T] = v.detach()
        self.length = start_pos + T
        return (
            self.cache_k[:, :, : self.length],
            self.cache_v[:, :, : self.length],
        )

    def reset(self) -> None:
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.length = 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Grouped / Multi Query Attention
# ─────────────────────────────────────────────────────────────────────────────
class GroupedQueryAttention(nn.Module):

    def __init__(
        self,
        embed_dim:     int,
        num_heads:     int,
        num_kv_heads:  int,
        max_seq_len:   int,
        attn_dropout:  float = 0.0,
        resid_dropout: float = 0.0,
        use_kv_cache:  bool  = False,
        rope_base:     int   = 10_000,
        cache_factor:  int   = 4,
        use_flash:     bool  = False,
        norm_eps:      float = 1e-6,
    ) -> None:
        super().__init__()

        assert embed_dim % num_heads    == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups   = num_heads // num_kv_heads   # GQA groups
        self.head_dim     = embed_dim // num_heads
        self.embed_dim    = embed_dim
        self.use_kv_cache = use_kv_cache
        self.attn_drop_p  = attn_dropout

        # ── Backend selection ─────────────────────────────────────────────────
        self.use_flash = (
            use_flash and HAS_FLASH_ATTN and torch.cuda.is_available()
        )
        self.use_sdpa = HAS_SDPA and not self.use_flash

        # ── Projections ───────────────────────────────────────────────────────
        kv_dim = num_kv_heads * self.head_dim
        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj   = nn.Linear(embed_dim, kv_dim,    bias=False)
        self.v_proj   = nn.Linear(embed_dim, kv_dim,    bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop  = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

        # ── Causal mask ───────────────────────────────────────────────────────
        causal = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal)

        # ── RoPE ──────────────────────────────────────────────────────────────
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, max_seq_len,
            base=rope_base, cache_factor=cache_factor,
        )

        self._kv_cache: Optional[KVCache] = None

    # ── KV cache management ───────────────────────────────────────────────────
    def _get_or_create_cache(
        self,
        B:      int,
        device: torch.device,
        dtype:  torch.dtype,
    ) -> KVCache:
        max_len = self.causal_mask.size(0)
        if (
            self._kv_cache is None
            or self._kv_cache.cache_k.size(0) != B
            or self._kv_cache.cache_k.device  != device
        ):
            self._kv_cache = KVCache(
                B, self.num_kv_heads, max_len,
                self.head_dim, device, dtype,
            )
        return self._kv_cache

    def reset_cache(self) -> None:
        if self._kv_cache is not None:
            self._kv_cache.reset()

    # ── Manual attention (fallback) ───────────────────────────────────────────
    def _manual_attn(
        self,
        q:         torch.Tensor,
        k:         torch.Tensor,
        v:         torch.Tensor,
        start_pos: int,
        T:         int,
    ) -> torch.Tensor:
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        seq_k  = k.size(2)
        # Slice causal mask for current positions
        mask   = self.causal_mask[start_pos : start_pos + T, :seq_k]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn   = F.softmax(scores.float(), dim=-1).to(q.dtype)
        attn   = self.attn_drop(attn)
        return torch.matmul(attn, v)

    def forward(
        self,
        x:         torch.Tensor,
        start_pos: int  = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # ── Project Q K V ─────────────────────────────────────────────────────
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ── RoPE ──────────────────────────────────────────────────────────────
        cos, sin = self.rotary_emb(start_pos + T)
        pos_ids  = torch.arange(start_pos, start_pos + T, device=x.device)
        q, k     = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)

        # ── KV Cache ──────────────────────────────────────────────────────────
        if use_cache and self.use_kv_cache:
            cache = self._get_or_create_cache(B, x.device, x.dtype)
            k, v  = cache.update(k, v, start_pos)

        # ── GQA/MQA: expand K,V heads to match Q heads ────────────────────────
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # ── Attention backend ─────────────────────────────────────────────────
        # Use fast paths only during training (no KV cache active)
        use_fast_path = not (use_cache and self.use_kv_cache)

        if self.use_flash and use_fast_path:
            # FlashAttention expects [B, T, H, D]
            out = flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p = self.attn_drop_p if self.training else 0.0,
                causal    = True,
            )
            out = out.contiguous().view(B, T, C)

        elif self.use_sdpa and use_fast_path:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = None,
                dropout_p = self.attn_drop_p if self.training else 0.0,
                is_causal = True,
            )
            out = out.transpose(1, 2).contiguous().view(B, T, C)

        else:
            # Manual fallback — handles KV cache case
            out = self._manual_attn(q, k, v, start_pos, T)
            out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_drop(self.out_proj(out))


# ─────────────────────────────────────────────────────────────────────────────
# 5. SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────
class SwiGLUFFN(nn.Module):

    def __init__(
        self,
        embed_dim:      int,
        ffn_multiplier: float = 2.6667,
        multiple_of:    int   = 64,
        dropout:        float = 0.0,
    ) -> None:
        super().__init__()
        raw        = int(embed_dim * ffn_multiplier)
        hidden_dim = ((raw + multiple_of - 1) // multiple_of) * multiple_of

        self.w1      = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2      = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.w3      = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.w2(F.silu(self.w1(x)) * self.w3(x))
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Decoder Block
# ─────────────────────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim:              int,
        num_heads:              int,
        num_kv_heads:           int,
        max_seq_len:            int,
        ffn_multiplier:         float = 2.6667,
        multiple_of:            int   = 64,
        dropout:                float = 0.0,
        attn_dropout:           float = 0.0,
        ffn_dropout:            float = 0.0,
        use_kv_cache:           bool  = False,
        rope_base:              int   = 10_000,
        cache_factor:           int   = 4,
        use_flash:              bool  = False,
        gradient_checkpointing: bool  = False,
        norm_eps:               float = 1e-6,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        self.norm1 = RMSNorm(embed_dim, eps=norm_eps)
        self.norm2 = RMSNorm(embed_dim, eps=norm_eps)

        self.attn = GroupedQueryAttention(
            embed_dim     = embed_dim,
            num_heads     = num_heads,
            num_kv_heads  = num_kv_heads,
            max_seq_len   = max_seq_len,
            attn_dropout  = attn_dropout,
            resid_dropout = dropout,
            use_kv_cache  = use_kv_cache,
            rope_base     = rope_base,
            cache_factor  = cache_factor,
            use_flash     = use_flash,
            norm_eps      = norm_eps,
        )
        self.ffn = SwiGLUFFN(
            embed_dim      = embed_dim,
            ffn_multiplier = ffn_multiplier,
            multiple_of    = multiple_of,
            dropout        = ffn_dropout,
        )

    def _attn_block(
        self,
        x:         torch.Tensor,
        start_pos: int,
        use_cache: bool,
    ) -> torch.Tensor:
        return x + self.attn(
            self.norm1(x), start_pos=start_pos, use_cache=use_cache
        )

    def _ffn_block(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm2(x))

    def forward(
        self,
        x:         torch.Tensor,
        start_pos: int  = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            x = checkpoint(
                lambda h: self._attn_block(h, start_pos, use_cache),
                x,
                use_reentrant=False,
            )
            x = checkpoint(self._ffn_block, x, use_reentrant=False)
        else:
            x = self._attn_block(x, start_pos, use_cache)
            x = self._ffn_block(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 7. SLM — Main Model
# ─────────────────────────────────────────────────────────────────────────────
class SLM(nn.Module):

    def __init__(self, config: SLMConfig) -> None:
        super().__init__()
        self.config       = config
        self.use_kv_cache = config.use_kv_cache

        # ── Embedding ─────────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(
            config.vocab_size, config.embed_dim,
            padding_idx=config.pad_id,
        )

        # ── Transformer layers ────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            DecoderBlock(
                embed_dim              = config.embed_dim,
                num_heads              = config.num_heads,
                num_kv_heads           = config.num_kv_heads,
                max_seq_len            = config.max_seq_len,
                ffn_multiplier         = config.ffn_multiplier,
                multiple_of            = config.multiple_of,
                dropout                = config.dropout,
                attn_dropout           = config.attn_dropout,
                ffn_dropout            = config.ffn_dropout,
                use_kv_cache           = config.use_kv_cache,
                rope_base              = config.rope_base,
                cache_factor           = config.cache_factor,
                use_flash              = config.use_flash,
                gradient_checkpointing = config.gradient_checkpointing,
                norm_eps               = config.norm_eps,
            )
            for _ in range(config.num_layers)
        ])

        # ── Output ────────────────────────────────────────────────────────────
        self.norm    = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(
            config.embed_dim, config.vocab_size, bias=False
        )

        # ── Weight init ───────────────────────────────────────────────────────
        self._init_weights(config.init_scale, config.num_layers)

        # ✅ Weight tying: share embedding and lm_head weights
        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        logger.info(
            "SLM | params=%s | %dL %dH(%dkv) %dD "
            "| vocab=%d ctx=%d | flash=%s sdpa=%s | mode=%s",
            f"{self.num_parameters():,}",
            config.num_layers, config.num_heads,
            config.num_kv_heads, config.embed_dim,
            config.vocab_size,   config.max_seq_len,
            "ON" if config.use_flash and HAS_FLASH_ATTN else "OFF",
            "ON" if HAS_SDPA else "OFF",
            "MQA" if config.num_kv_heads == 1
            else ("MHA" if config.num_kv_heads == config.num_heads
                  else f"GQA(g={config.num_heads // config.num_kv_heads})"),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Init
    # ─────────────────────────────────────────────────────────────────────────
    def _init_weights(
        self, init_scale: float, num_layers: int
    ) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_scale)

        
        residual_std = init_scale / math.sqrt(2.0 * num_layers)
        for layer in self.layers:
            nn.init.normal_(
                layer.attn.out_proj.weight, mean=0.0, std=residual_std
            )
            nn.init.normal_(
                layer.ffn.w2.weight, mean=0.0, std=residual_std
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Properties / utils
    # ─────────────────────────────────────────────────────────────────────────
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if not trainable_only or p.requires_grad
        )

    def reset_kv_cache(self) -> None:
        for layer in self.layers:
            layer.attn.reset_cache()

    def model_info(self) -> Dict:
        info = asdict(self.config)
        info["total_params"]     = self.num_parameters(trainable_only=False)
        info["trainable_params"] = self.num_parameters(trainable_only=True)
        info["has_flash_attn"]   = HAS_FLASH_ATTN
        info["has_sdpa"]         = HAS_SDPA
        info["attn_mode"] = (
            "MQA" if self.config.num_kv_heads == 1
            else ("MHA" if self.config.num_kv_heads == self.config.num_heads
                  else f"GQA(groups={self.config.num_heads // self.config.num_kv_heads})")
        )
        return info

    def compile(self, **kwargs) -> "SLM":
        if not hasattr(torch, "compile"):
            logger.warning(
                "torch.compile requires PyTorch >= 2.0 — skipped."
            )
            return self
        compiled = torch.compile(self, **kwargs)
        logger.info("torch.compile applied")
        return compiled

    # ─────────────────────────────────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(
            os.path.dirname(os.path.abspath(path)), exist_ok=True
        )
        torch.save(
            {
                "config":      asdict(self.config),
                "model_state": self.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved -> %s", path)

    @classmethod
    def load(
        cls,
        path:         str,
        map_location: Optional[str] = None,
        override:     Optional[dict] = None,
    ) -> "SLM":
        ckpt  = torch.load(
            path,
            map_location = map_location or "cpu",
            weights_only = False,
        )
        cfg_d = ckpt["config"]
        if override:
            cfg_d.update(override)
        
        #    SLM is an nn.Module not a dataclass — cls.__dataclass_fields__ crashes
        valid_keys   = {f.name for f in fields(SLMConfig)}
        filtered_cfg = {k: v for k, v in cfg_d.items() if k in valid_keys}
        model        = cls(SLMConfig(**filtered_cfg))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        logger.info(
            "Checkpoint loaded <- %s  (params=%s)",
            path, f"{model.num_parameters():,}",
        )
        return model

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids:       torch.Tensor,
        targets:         Optional[torch.Tensor] = None,
        loss_mask:       Optional[torch.Tensor] = None,
        label_smoothing: float                  = 0.0,
        start_pos:       int                    = 0,
        use_cache:       bool                   = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} > max_seq_len {self.config.max_seq_len}.\n"
            f"Reduce block_size in config.yaml."
        )

        x = self.token_emb(input_ids)

        for layer in self.layers:
            x = layer(x, start_pos=start_pos, use_cache=use_cache)

        x      = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        acc  = 0.0

        if targets is not None:
            flat_logits  = logits.view(-1, self.config.vocab_size)
            flat_targets = targets.view(-1)

            if loss_mask is not None:
                flat_mask    = loss_mask.view(-1)
                flat_logits  = flat_logits[flat_mask]
                flat_targets = flat_targets[flat_mask]

            if flat_targets.numel() > 0:
                loss = F.cross_entropy(
                    flat_logits,
                    flat_targets,
                    label_smoothing = label_smoothing,
                    ignore_index    = -1,
                )

                with torch.no_grad():
                    preds   = flat_logits.argmax(-1)
                    valid   = flat_targets != -1
                    correct = (preds[valid] == flat_targets[valid]).float()
                    acc     = correct.mean().item() if valid.sum() > 0 else 0.0
            else:
                loss = torch.tensor(
                    0.0, device=logits.device, requires_grad=True
                )

        return logits, loss, acc

    # ─────────────────────────────────────────────────────────────────────────
    # Embeddings
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        pooling:   Literal["mean", "last", "max"] = "mean",
    ) -> torch.Tensor:
        
        was_training = self.training
        self.eval()

        x = self.token_emb(input_ids)
        for layer in self.layers:
            x = layer(x, start_pos=0, use_cache=False)
        x = self.norm(x)

        if pooling == "mean":
            mask = (input_ids != self.config.pad_id).unsqueeze(-1).float()
            emb  = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        elif pooling == "last":
            lengths = (input_ids != self.config.pad_id).sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            emb     = x[
                torch.arange(x.size(0), device=x.device), lengths
            ]
        elif pooling == "max":
            # Mask PAD positions with -inf before max
            mask = (input_ids == self.config.pad_id).unsqueeze(-1)
            x_   = x.masked_fill(mask, float("-inf"))
            emb  = x_.max(dim=1).values
        else:
            raise ValueError(
                f"Unknown pooling: '{pooling}'. "
                "Use 'mean' | 'last' | 'max'"
            )

        emb = F.normalize(emb, p=2, dim=-1)

        if was_training:
            self.train()
        return emb

    # ─────────────────────────────────────────────────────────────────────────
    # Sampling helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _sample(
        logits:      torch.Tensor,
        temperature: float,
        top_k:       Optional[int],
        top_p:       Optional[float],
    ) -> torch.Tensor:
        
        if temperature <= 0.0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        # ── Top-k ─────────────────────────────────────────────────────────────
        if top_k is not None and top_k > 0:
            k         = min(top_k, logits.size(-1))
            kth_vals  = torch.topk(logits, k, dim=-1).values[:, -1:]
            logits    = logits.masked_fill(logits < kth_vals, float("-inf"))

        
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(
                logits, descending=True, dim=-1
            )
            sorted_probs  = F.softmax(sorted_logits, dim=-1)
            cum_probs     = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_remove = (cum_probs - sorted_probs) >= top_p
           
            sorted_remove[:, 0] = False

            sorted_logits = sorted_logits.masked_fill(
                sorted_remove, float("-inf")
            )
            logits = torch.zeros_like(logits).scatter(
                1, sorted_idx, sorted_logits
            )

        probs = F.softmax(logits, dim=-1).clamp(min=0.0)
        
        row_sum = probs.sum(dim=-1, keepdim=True)
        probs   = torch.where(
            row_sum > 0,
            probs / row_sum,
            torch.full_like(probs, 1.0 / probs.size(-1)),
        )

        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def _apply_repetition_penalty(
        logits:             torch.Tensor,
        generated_ids:      torch.Tensor,
        repetition_penalty: float,
        repetition_window:  int,
    ) -> torch.Tensor:
        
        if repetition_penalty == 1.0:
            return logits
        
        logits = logits.clone()

        window = generated_ids
        if window.size(1) > repetition_window:
            window = window[:, -repetition_window:]
        
        for b in range(logits.size(0)):
            unique_ids = window[b].unique()
            score      = logits[b, unique_ids]
            logits[b, unique_ids] = torch.where(
                score > 0,
                score / repetition_penalty,
                score * repetition_penalty,
            )
        return logits

    # ─────────────────────────────────────────────────────────────────────────
    # Single-sequence generation
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        idx:                torch.Tensor,
        max_new_tokens:     int            = 256,
        temperature:        float          = 1.0,
        top_k:              Optional[int]  = 50,
        top_p:              Optional[float]= 0.9,
        repetition_penalty: float          = 1.1,
        repetition_window:  int            = 64,
        eos_id:             Optional[int]  = None,
        pad_id:             Optional[int]  = None,
    ) -> torch.Tensor:
        
        self.eval()
        self.reset_kv_cache()

        B       = idx.size(0)
        done    = torch.zeros(B, dtype=torch.bool, device=idx.device)
        _pad_id = pad_id if pad_id is not None else self.config.pad_id

        for step in range(max_new_tokens):
            if step == 0:
                cur_input = idx
                start_pos = 0
            else:
                cur_input = idx[:, -1:]
                
                start_pos = idx.size(1) - 1

            logits, _, _ = self.forward(
                cur_input,
                start_pos = start_pos,
                use_cache = self.use_kv_cache,
            )

            next_logits = logits[:, -1, :]
            next_logits = self._apply_repetition_penalty(
                next_logits, idx,
                repetition_penalty, repetition_window,
            )
            next_token = self._sample(
                next_logits, temperature, top_k, top_p
            )

            # Fill completed sequences with pad
            next_token = torch.where(
                done.unsqueeze(1),
                torch.full_like(next_token, _pad_id),
                next_token,
            )

            idx = torch.cat([idx, next_token], dim=1)

            if eos_id is not None:
                done = done | (next_token.squeeze(1) == eos_id)
                if done.all():
                    break

        return idx

    # ─────────────────────────────────────────────────────────────────────────
    # ✅ NEW: Multi-query batch generation
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def batch_generate(
        self,
        prompts:            List[torch.Tensor],
        max_new_tokens:     int            = 256,
        temperature:        float          = 1.0,
        top_k:              Optional[int]  = 50,
        top_p:              Optional[float]= 0.9,
        repetition_penalty: float          = 1.1,
        repetition_window:  int            = 64,
        eos_id:             Optional[int]  = None,
    ) -> List[torch.Tensor]:
        
        if not prompts:
            return []

        device  = prompts[0].device
        _pad_id = self.config.pad_id

        # ── Pad all prompts to max prompt length (left-pad) ───────────────────
        max_len       = max(p.size(1) for p in prompts)
        prompt_lens   = [p.size(1) for p in prompts]
        padded        = torch.full(
            (len(prompts), max_len),
            fill_value = _pad_id,
            dtype      = torch.long,
            device     = device,
        )
        for i, p in enumerate(prompts):
            # Right-align: pad on left
            padded[i, max_len - p.size(1) :] = p[0]

        # ── Generate as a single batch ────────────────────────────────────────
        full_output = self.generate(
            idx                = padded,
            max_new_tokens     = max_new_tokens,
            temperature        = temperature,
            top_k              = top_k,
            top_p              = top_p,
            repetition_penalty = repetition_penalty,
            repetition_window  = repetition_window,
            eos_id             = eos_id,
            pad_id             = _pad_id,
        )

        # ── Strip left-padding from each output ───────────────────────────────
        results = []
        for i, orig_len in enumerate(prompt_lens):
            pad_added = max_len - orig_len
            results.append(full_output[i : i + 1, pad_added:])

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Streaming generation
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def stream_generate(
        self,
        idx:                torch.Tensor,
        max_new_tokens:     int            = 256,
        temperature:        float          = 1.0,
        top_k:              Optional[int]  = 50,
        top_p:              Optional[float]= 0.9,
        repetition_penalty: float          = 1.1,
        repetition_window:  int            = 64,
        eos_id:             Optional[int]  = None,
    ) -> Generator[int, None, None]:
        
        assert idx.size(0) == 1, \
            "stream_generate only supports batch_size=1. " \
            "Use generate() for batch inference."

        self.eval()
        self.reset_kv_cache()

        for step in range(max_new_tokens):
            if step == 0:
                cur_input = idx
                start_pos = 0
            else:
                cur_input = idx[:, -1:]
                start_pos = idx.size(1) - 1

            logits, _, _ = self.forward(
                cur_input,
                start_pos = start_pos,
                use_cache = self.use_kv_cache,
            )

            next_logits = logits[:, -1, :]
            next_logits = self._apply_repetition_penalty(
                next_logits, idx,
                repetition_penalty, repetition_window,
            )
            next_token = self._sample(
                next_logits, temperature, top_k, top_p
            )

            tok_id = next_token.item()
            idx    = torch.cat([idx, next_token], dim=1)

            yield tok_id

            if eos_id is not None and tok_id == eos_id:
                break

    # ─────────────────────────────────────────────────────────────────────────
    # Repr
    # ─────────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        c     = self.config
        p     = self.num_parameters(trainable_only=False)
        tp    = self.num_parameters(trainable_only=True)
        ffn_h = (
            (int(c.embed_dim * c.ffn_multiplier) + c.multiple_of - 1)
            // c.multiple_of
        ) * c.multiple_of
        mode  = (
            "MQA" if c.num_kv_heads == 1
            else ("MHA" if c.num_kv_heads == c.num_heads
                  else f"GQA(groups={c.num_heads // c.num_kv_heads})")
        )
        return (
            f"SLM(\n"
            f"  total_params     = {p:,}\n"
            f"  trainable_params = {tp:,}\n"
            f"  layers           = {c.num_layers}\n"
            f"  attention        = {mode}  "
            f"{c.num_heads}Q / {c.num_kv_heads}KV\n"
            f"  embed_dim        = {c.embed_dim}\n"
            f"  ffn_hidden       = {ffn_h}\n"
            f"  vocab_size       = {c.vocab_size}\n"
            f"  max_seq_len      = {c.max_seq_len}\n"
            f"  rope_base        = {c.rope_base}\n"
            f"  dropout          = {c.dropout}\n"
            f"  tie_weights      = {c.tie_weights}\n"
            f"  flash_attn       = "
            f"{'ON' if c.use_flash and HAS_FLASH_ATTN else 'OFF'}\n"
            f"  sdpa             = {'ON' if HAS_SDPA else 'OFF'}\n"
            f"  grad_ckpt        = "
            f"{'ON' if c.gradient_checkpointing else 'OFF'}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Model factory
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS: Dict[str, dict] = {
    #  name        embed  layers  Q-heads  KV-heads  ctx
    "slm-6m":   dict(embed_dim=256,  num_layers=6,  num_heads=8,  num_kv_heads=2,  max_seq_len=256,  ffn_multiplier=2.6667, multiple_of=64),
    "slm-13m":  dict(embed_dim=384,  num_layers=8,  num_heads=8,  num_kv_heads=2,  max_seq_len=256,  ffn_multiplier=2.6667, multiple_of=64),
    "slm-15m":  dict(embed_dim=256,  num_layers=6,  num_heads=8,  num_kv_heads=2,  max_seq_len=512,  ffn_multiplier=2.6667, multiple_of=64),
    "slm-42m":  dict(embed_dim=512,  num_layers=8,  num_heads=8,  num_kv_heads=2,  max_seq_len=1024, ffn_multiplier=2.6667, multiple_of=64),
    "slm-117m": dict(embed_dim=768,  num_layers=12, num_heads=12, num_kv_heads=4,  max_seq_len=1024, ffn_multiplier=2.6667, multiple_of=64),
    "slm-350m": dict(embed_dim=1024, num_layers=24, num_heads=16, num_kv_heads=4,  max_seq_len=2048, ffn_multiplier=2.6667, multiple_of=64),
    "slm-mqa":  dict(embed_dim=512,  num_layers=8,  num_heads=8,  num_kv_heads=1,  max_seq_len=1024, ffn_multiplier=2.6667, multiple_of=64),
    "slm-mha":  dict(embed_dim=512,  num_layers=8,  num_heads=8,  num_kv_heads=8,  max_seq_len=1024, ffn_multiplier=2.6667, multiple_of=64),
}


def create_model(
    model_size: str,
    vocab_size: int,
    **kwargs,
) -> SLM:
    
    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_size '{model_size}'.\n"
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )
    cfg_dict = {
        **MODEL_CONFIGS[model_size],
        "vocab_size": vocab_size,
        **kwargs,
    }
    return SLM(SLMConfig(**cfg_dict))
