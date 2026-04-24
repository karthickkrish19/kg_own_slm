import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Config:
    # Paths
    corpus_files:          List[str]       = field(default_factory=lambda: ["data/input.txt"])
    tokenizer_file:        str             = "out/tokenizer.json"
    tokenized_cache_dir:   Optional[str]   = "out/tokenized_cache"
    model_save_dir:        str             = "out/checkpoints"
    log_dir:               str             = "out/logs"
    embeddings_save_path:  Optional[str]   = "out/embeddings.pt"

    # Tokenizer
    vocab_size:                  int        = 50000
    min_frequency:               int        = 2
    special_tokens:              List[str]  = field(
        default_factory=lambda: ["<pad>", "<s>", "</s>", "<unk>"]
    )
    tokenizer_normalization:     str        = "nfd"
    add_bos:                     bool       = True
    add_eos:                     bool       = True
    stop_at_eos:                 bool       = True
    vocab_warning_threshold:     float      = 0.9

    # Data
    block_size:              int            = 256
    train_stride:            int            = 128
    tokenizer_corpus_files:  Optional[List[str]] = None
    val_split:               float          = 0.05
    streaming:               bool           = False
    shuffle_buffer:          int            = 10000
    steps_per_epoch:         Optional[int]  = None
    deduplicate_text:        bool           = False
    dedup_bloom_error_rate:  float          = 0.01

    # Training
    batch_size:          int            = 16
    epochs:              int            = 10
    grad_accum_steps:    int            = 2
    learning_rate:       float          = 3e-4
    min_lr:              float          = 3e-5
    lr_schedule:         bool           = True
    warmup_steps:        int            = 200
    weight_decay:        float          = 0.1
    no_decay_params:     List[str]      = field(
        default_factory=lambda: ["bias", "norm.weight", "norm1.weight", "norm2.weight"]
    )
    label_smoothing:     float          = 0.1
    gradient_clip:       float          = 1.0
    log_every:           int            = 100
    eval_every:          int            = 1000
    save_every:          int            = 2000
    keep_checkpoints:    int            = 3
    early_stop_patience: Optional[int]  = 5

    # Model
    embed_dim:              int            = 384
    num_heads:              int            = 8
    num_kv_heads:           int            = 2
    num_layers:             int            = 8
    head_dim:               Optional[int]  = None
    dropout:                float          = 0.1
    attn_dropout:           float          = 0.05
    ffn_dropout:            float          = 0.1
    norm_eps:               float          = 1e-6
    init_scale:             float          = 0.02
    use_kv_cache:           bool           = True
    rope_base:              int            = 10000
    cache_factor:           int            = 4
    use_flash:              bool           = False          
    compile_model:          bool           = False
    tie_weights:            bool           = True
    ffn_multiplier:         float          = 2.6667
    multiple_of:            int            = 64
    gradient_checkpointing: bool           = False
    model_size:             str            = "slm-6m"

    # Generation
    max_new_tokens:      int    = 200
    temperature:         float  = 0.8
    top_k:               int    = 40
    top_p:               float  = 0.92
    repetition_penalty:  float  = 1.3
    repetition_window:   int    = 64

    # Safety
    enable_safety_filter: bool      = True
    safety_keywords:      List[str] = field(default_factory=lambda: [
        "bomb", "kill", "hack", "exploit", "malware", "suicide"
    ])

    # Hardware
    seed:              int   = 42
    distributed:       bool  = False
    mixed_precision:   str   = "auto"
    num_workers:       int   = 0
    pin_memory:        bool  = False

    # Logging
    use_tensorboard:  bool           = True
    use_wandb:        bool           = False
    wandb_project:    str            = "slm_project"
    wandb_entity:     Optional[str]  = None

    # RAG
    rag_enabled:               bool  = False
    rag_vector_store_path:     str   = "out/vector_store"
    rag_top_k:                 int   = 5
    rag_chunk_size:            int   = 500
    rag_chunk_overlap:         int   = 100
    rag_use_web_search:        bool  = False
    rag_web_search_num_results:int   = 3
    rag_web_search_max_chars:  int   = 2000
    rag_embedding_layer:       str   = "mean"
    rag_embedding_normalize:   bool  = True
    rag_embedding_batch_size:  int   = 32
    rag_embedding_projection:  bool  = False
    rag_cache_size:            int   = 1000

    # API
    api_host:              str            = "0.0.0.0"
    api_port:              int            = 8000
    api_key:               Optional[str]  = None
    rate_limit_per_minute: int            = 60

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        valid    = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    def to_yaml(self, path: str) -> None:
        dirp = os.path.dirname(path)
        if dirp:
            os.makedirs(dirp, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, sort_keys=False)

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim {self.embed_dim} not divisible by num_heads {self.num_heads}"
        assert self.block_size > 0,          "block_size must be > 0"
        assert 0.0 <= self.val_split < 1.0,  "val_split must be in [0, 1)"
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads {self.num_heads} must be divisible by num_kv_heads {self.num_kv_heads}"
