"""
SLM RAG Pipeline
================
Retrieval-Augmented Generation with vector store, model-based embeddings,
LRU embedding cache, and optional web search.

Features:
  - ModelEmbedder: uses SLM hidden states for dense embeddings
  - EmbeddingCache: thread-safe LRU cache
  - VectorStore: FAISS-backed (flat/IVF/HNSW), with fallback to numpy
  - WebSearch: DuckDuckGo HTML scraping with rate limiting
  - RAGPipeline: ties everything together

Usage:
    from rag import RAGPipeline
    rag = RAGPipeline(config, model, tokenizer, device)
    rag.ingest_documents(["doc1 text", "doc2 text"])
    context = rag.get_context("What is geometry?")
"""

import os
import re
import json
import time
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Cache (LRU, thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
class EmbeddingCache:
    def __init__(self, capacity: int = 1000) -> None:
        self.cache = OrderedDict()
        self.capacity = capacity
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        return None

    def put(self, key: str, emb) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)
            self.cache[key] = emb


# ─────────────────────────────────────────────────────────────────────────────
# Text chunker
# ─────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Model Embedder
# ─────────────────────────────────────────────────────────────────────────────
class ModelEmbedder:
    """Produces dense embeddings using SLM hidden states."""

    def __init__(self, model, tokenizer, device, config) -> None:
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for RAG. pip install numpy")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pooling = config.rag_embedding_layer
        self.normalize = config.rag_embedding_normalize
        self.batch_size = config.rag_embedding_batch_size
        self.cache = EmbeddingCache(config.rag_cache_size)
        self.model.eval()

    def _compute_batch(self, texts: List[str]) -> "np.ndarray":
        all_embs = []
        seq_len = self.model.config.max_seq_len
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # W-3: must match training — model was trained without BOS/EOS
            encoded = self.tokenizer.batch_encode(
                batch, max_len=seq_len, pad=True, truncate=True,
                add_special_tokens=False,
            )
            ids_list = encoded["input_ids"]
            inp = torch.tensor(ids_list, dtype=torch.long, device=self.device)
            with torch.no_grad():
                emb = self.model.get_embeddings(inp, pooling=self.pooling)
            all_embs.append(emb.cpu().float().numpy())
        result = np.vstack(all_embs)
        if self.normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / (norms + 1e-8)
        return result

    def embed(self, texts: List[str]) -> "np.ndarray":
        results = [None] * len(texts)
        uncached_idx = []
        uncached_texts = []
        for i, t in enumerate(texts):
            cached = self.cache.get(t)
            if cached is not None:
                results[i] = cached
            else:
                uncached_idx.append(i)
                uncached_texts.append(t)
        if uncached_texts:
            new_embs = self._compute_batch(uncached_texts)
            for j, idx in enumerate(uncached_idx):
                self.cache.put(texts[idx], new_embs[j])
                results[idx] = new_embs[j]
        return np.stack(results)


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store
# ─────────────────────────────────────────────────────────────────────────────
class VectorStore:
    """FAISS-backed vector store with numpy fallback."""

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        path: Optional[str] = None,
    ) -> None:
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for VectorStore")
        self.dim = embedding_dim
        self.index_type = index_type
        self.chunks: List[str] = []
        self.metadata: List[dict] = []
        self.path = path
        self.lock = threading.Lock()

        # Trigger load when meta file exists (covers both FAISS and numpy stores)
        if path and Path(path + ".meta").exists():
            self._load(path)
        else:
            self._create_index()

    def _create_index(self) -> None:
        # W-1: always initialize so attribute exists regardless of FAISS
        self._numpy_store = None
        self.index = None
        if not FAISS_AVAILABLE:
            return
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, 100)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    def add_chunks(
        self,
        chunks: List[str],
        embeddings: "np.ndarray",
        metadata: Optional[List[dict]] = None,
    ) -> None:
        with self.lock:
            embeddings = embeddings.astype(np.float32)

            if FAISS_AVAILABLE and self.index is not None:
                if hasattr(self.index, "is_trained") and not self.index.is_trained:
                    if len(embeddings) >= 100:
                        self.index.train(embeddings)
                    else:
                        self.index = faiss.IndexFlatIP(self.dim)
                self.index.add(embeddings)
            else:
                # Numpy fallback
                if self._numpy_store is None:
                    self._numpy_store = embeddings
                else:
                    self._numpy_store = np.vstack(
                        [self._numpy_store, embeddings]
                    )

            self.chunks.extend(chunks)
            self.metadata.extend(metadata or [{}] * len(chunks))

    def search(
        self, query_emb: "np.ndarray", k: int = 5
    ) -> List[Tuple[str, float, dict]]:
        if not self.chunks:
            return []

        with self.lock:
            k = min(k, len(self.chunks))
            query = query_emb.astype(np.float32).reshape(1, -1)

            if FAISS_AVAILABLE and self.index is not None:
                scores_2d, indices_2d = self.index.search(query, k)
                scores_flat  = scores_2d[0]
                indices_flat = indices_2d[0]
            elif self._numpy_store is not None:
                sims = (self._numpy_store @ query.T).squeeze()
                # C-2: atleast_1d guards against 0-d scalar when store has 1 vector
                sims         = np.atleast_1d(sims)
                indices_flat = np.argsort(-sims)[:k]
                scores_flat  = sims[indices_flat]
            else:
                return []

            results = []
            for score, idx in zip(scores_flat, indices_flat):
                if 0 <= idx < len(self.chunks):
                    results.append(
                        (self.chunks[idx], float(score), self.metadata[idx])
                    )
            return results

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, path + ".index")
        elif self._numpy_store is not None:
            # C-1: persist numpy fallback so vectors survive a server restart
            np.save(path + ".npy", self._numpy_store)
        with open(path + ".meta", "w", encoding="utf-8") as f:
            json.dump(
                {"chunks": self.chunks, "metadata": self.metadata}, f,
                ensure_ascii=False,
            )
        logger.info("VectorStore saved -> %s (%d chunks)", path, len(self.chunks))

    def _load(self, path: str) -> None:
        # W-2 / C-1: two-phase load so meta failure vs index failure are handled
        # independently; successful FAISS index is never silently discarded.

        # Phase 1 — metadata (chunks + meta JSON) — must succeed to continue
        try:
            with open(path + ".meta", "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks   = data["chunks"]
            self.metadata = data["metadata"]
        except Exception as e:
            logger.warning(
                "VectorStore meta load failed (%s) — creating fresh store.", e
            )
            self._create_index()
            return

        # Phase 2 — vector index — tolerates failure; chunks are already loaded
        try:
            if FAISS_AVAILABLE and Path(path + ".index").exists():
                self.index = faiss.read_index(path + ".index")
            elif Path(path + ".npy").exists():
                # C-1: load persisted numpy fallback store
                self._numpy_store = np.load(path + ".npy")
            else:
                logger.warning(
                    "VectorStore: no vector index found at '%s' — "
                    "chunks loaded but retrieval disabled until re-ingestion.",
                    path,
                )
        except Exception as e:
            logger.warning(
                "VectorStore index load failed (%s) — fresh index "
                "(chunks from meta are intact; re-ingest to rebuild vectors).", e
            )
            self._create_index()

        logger.info(
            "VectorStore loaded <- %s (%d chunks)", path, len(self.chunks)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Web Search (DuckDuckGo HTML)
# ─────────────────────────────────────────────────────────────────────────────
class WebSearch:
    """Rate-limited DuckDuckGo search with page text extraction."""

    def __init__(self, num_results: int = 3, max_chars: int = 2000) -> None:
        self.num_results = num_results
        self.max_chars = max_chars
        self._last_req = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_req
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_req = time.time()

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", "", text)
        return text.strip()

    def _extract_text(self, html: str) -> str:
        if not WEB_AVAILABLE:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in [
            "script", "style", "nav", "footer", "header", "form", "aside", "iframe"
        ]:
            for elem in soup.find_all(tag):
                elem.decompose()
        for sel in [
            "article", "main", '[role="main"]', ".content",
            "#content", ".post-content",
        ]:
            elem = soup.select_one(sel)
            if elem:
                return self._clean_text(elem.get_text(separator=" "))
        body = soup.find("body")
        if body:
            return self._clean_text(body.get_text(separator=" "))
        return self._clean_text(soup.get_text(separator=" "))

    def _ddg_search(self, query: str) -> List[dict]:
        if not WEB_AVAILABLE:
            return []
        self._rate_limit()
        try:
            url = "https://html.duckduckgo.com/html/"
            resp = requests.get(
                url,
                params={"q": query, "kl": "us-en"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10.0,
            )
            resp.raise_for_status()
        except Exception:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result in soup.select(".result"):
            a_tag = result.select_one(".result__title a")
            if not a_tag:
                continue
            title = self._clean_text(a_tag.get_text())
            link = a_tag.get("href", "")
            if not link or link.startswith("//duckduckgo"):
                continue
            snippet_tag = result.select_one(".result__snippet")
            snippet = (
                self._clean_text(snippet_tag.get_text()) if snippet_tag else ""
            )
            if len(snippet) >= 30:
                results.append(
                    {"title": title, "url": link, "snippet": snippet}
                )
            if len(results) >= self.num_results * 2:
                break
        return results[: self.num_results * 2]

    def _fetch_page(self, url: str) -> str:
        if not WEB_AVAILABLE:
            return ""
        # Validate URL to prevent SSRF
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return ""
            if not parsed.hostname:
                return ""
            # Block private/internal network ranges
            hostname = parsed.hostname.lower()
            if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
                return ""
            if hostname.startswith("10.") or hostname.startswith("192.168."):
                return ""
            if hostname.startswith("169.254."):
                return ""
            if hostname.startswith("172."):
                parts = hostname.split(".")
                if len(parts) >= 2 and 16 <= int(parts[1]) <= 31:
                    return ""
        except Exception:
            return ""
        try:
            self._rate_limit()
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10.0,
                allow_redirects=False,
            )
            resp.raise_for_status()
            return self._extract_text(resp.text)
        except Exception:
            return ""

    def search(self, query: str) -> List[str]:
        if not WEB_AVAILABLE:
            logger.warning("Web search unavailable — pip install requests beautifulsoup4")
            return []
        results = self._ddg_search(query)
        if not results:
            return []
        output = []
        for r in results[: self.num_results]:
            snippet = r["snippet"]
            if len(snippet) < 200:
                page_text = self._fetch_page(r["url"])
                if len(page_text) > len(snippet):
                    snippet = page_text
            piece = f"[{r['title']}]\n{snippet}"
            if len(piece) > self.max_chars:
                piece = piece[: self.max_chars] + "..."
            if piece.strip():
                output.append(piece)
        return output


# ─────────────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class RAGPipeline:
    """Full RAG pipeline: ingest, embed, retrieve, optionally web-augment."""

    def __init__(self, config, model, tokenizer, device) -> None:
        self.config = config
        self.embedder = ModelEmbedder(model, tokenizer, device, config)
        self.vector_store = VectorStore(
            config.embed_dim, "flat", config.rag_vector_store_path
        )
        self.web_search = (
            WebSearch(
                config.rag_web_search_num_results,
                config.rag_web_search_max_chars,
            )
            if config.rag_use_web_search
            else None
        )
        logger.info(
            "RAGPipeline ready | dim=%d | web=%s",
            config.embed_dim,
            "enabled" if self.web_search else "disabled",
        )

    def retrieve(
        self, query: str, k: Optional[int] = None
    ) -> List[Tuple[str, float, dict]]:
        k = k or self.config.rag_top_k
        query_emb = self.embedder.embed([query])[0]
        return self.vector_store.search(query_emb, k)

    def get_context(self, query: str, k: Optional[int] = None) -> str:
        results = self.retrieve(query, k)
        chunks = [chunk for chunk, score, meta in results]

        # Optionally augment with web results
        if self.web_search and len(chunks) < 2:
            web_results = self.web_search.search(query)
            chunks.extend(web_results)

        if not chunks:
            return ""
        return "\n\n".join(chunks)

    def ingest_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> int:
        all_chunks = []
        all_meta = []
        for i, txt in enumerate(texts):
            chunks = chunk_text(
                txt, self.config.rag_chunk_size, self.config.rag_chunk_overlap
            )
            all_chunks.extend(chunks)
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            all_meta.extend([meta] * len(chunks))
        if not all_chunks:
            return 0
        embeddings = self.embedder.embed(all_chunks)
        self.vector_store.add_chunks(all_chunks, embeddings, all_meta)
        self.save()
        logger.info("Ingested %d chunks from %d documents", len(all_chunks), len(texts))
        return len(all_chunks)

    def ingest_file(self, file_path: str, metadata: Optional[dict] = None) -> int:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.ingest_documents([text], [metadata or {"source": str(path)}])

    def save(self) -> None:
        self.vector_store.save(self.config.rag_vector_store_path)
