"""
SLM API Server (FastAPI)
========================
REST API for text generation, streaming, embeddings, and RAG.

Endpoints:
  GET  /health          → health check
  POST /generate        → text generation (batch or stream)
  POST /embed           → dense embeddings
  POST /rag/ingest      → ingest documents into vector store
  POST /rag/query       → RAG-augmented generation

Usage:
  python api_server.py --config config.yaml --checkpoint out/checkpoints/best.pt
"""

import sys
import time
import hmac
import logging
import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def run_api(config: "Config", checkpoint_path: str) -> None:
    """Start the FastAPI server."""
    try:
        from contextlib import asynccontextmanager
        from fastapi import FastAPI, HTTPException, Request, Depends
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from fastapi.security import APIKeyHeader
        from pydantic import BaseModel, Field
        import uvicorn
    except ImportError:
        print(
            "FastAPI/uvicorn not installed.\n"
            "Run: pip install fastapi uvicorn"
        )
        sys.exit(1)

    engine = None
    rag_pipeline = None

    # ── Rate limiter (in-memory, per-IP) ──────────────────────────────────────
    _rate_buckets: dict = defaultdict(list)

    def _check_rate_limit(request: Request) -> None:
        limit = config.rate_limit_per_minute
        if limit <= 0:
            return
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = _rate_buckets[client_ip]
        # Purge entries older than 60s
        _rate_buckets[client_ip] = [t for t in window if now - t < 60.0]
        if not _rate_buckets[client_ip]:
            del _rate_buckets[client_ip]
            return
        if len(_rate_buckets[client_ip]) >= limit:
            raise HTTPException(429, "Rate limit exceeded")
        _rate_buckets[client_ip].append(now)

    # ── API key authentication ────────────────────────────────────────────────
    _api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def _verify_api_key(
        api_key: Optional[str] = Depends(_api_key_header),
    ) -> None:
        if config.api_key is None:
            return  # No key configured → open access
        if api_key is None:
            raise HTTPException(401, "Missing API key")
        if not hmac.compare_digest(api_key, config.api_key):
            raise HTTPException(403, "Invalid API key")

    # ── Lifespan (replaces deprecated @app.on_event("startup")) ──────────────
    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        nonlocal engine, rag_pipeline
        from inference import SLMInference

        engine = SLMInference(config, checkpoint_path)
        logger.info("API engine loaded.")

        if config.rag_enabled:
            from rag import RAGPipeline

            rag_pipeline = RAGPipeline(
                config,
                engine.model,
                engine.tok,
                engine.device,
            )
            logger.info("RAG pipeline loaded.")
        yield

    app = FastAPI(
        title="SLM API",
        version="1.0",
        description="Small Language Model — Production API",
        lifespan=lifespan,
    )

    # ── CORS — configurable origins ───────────────────────────────────────────
    cors_origins = getattr(config, "api_cors_origins", ["http://localhost:3000"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    # ── Request / Response models ─────────────────────────────────────────────
    class GenerateRequest(BaseModel):
        prompt: str = Field(..., min_length=1, max_length=10000)
        max_new_tokens: int = Field(200, ge=1, le=2048)
        temperature: float = Field(0.8, ge=0.0, le=2.0)
        top_k: int = Field(40, ge=0, le=500)
        top_p: float = Field(0.9, ge=0.0, le=1.0)
        repetition_penalty: float = Field(1.1, ge=1.0, le=3.0)
        stream: bool = False
        context: Optional[str] = None

    class GenerateResponse(BaseModel):
        text: str
        tokens_used: int
        elapsed_ms: float

    class EmbedRequest(BaseModel):
        texts: List[str] = Field(..., min_length=1)
        pooling: str = Field("mean", pattern="^(mean|last|max)$")

    class EmbedResponse(BaseModel):
        embeddings: List[List[float]]
        dim: int

    class RAGIngestRequest(BaseModel):
        texts: List[str] = Field(..., min_length=1)

    class RAGQueryRequest(BaseModel):
        query: str = Field(..., min_length=1)
        max_new_tokens: int = Field(200, ge=1, le=2048)
        temperature: float = Field(0.8, ge=0.0, le=2.0)
        top_k_docs: int = Field(5, ge=1, le=20)

    class RAGQueryResponse(BaseModel):
        answer: str
        context_used: str
        elapsed_ms: float

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        return {
            "status": "ok" if engine else "no_model",
            "rag_enabled": rag_pipeline is not None,
        }

    # ── Generate ──────────────────────────────────────────────────────────────
    @app.post("/generate", dependencies=[Depends(_verify_api_key)])
    async def generate_endpoint(req: GenerateRequest, request: Request):
        _check_rate_limit(request)
        if engine is None:
            raise HTTPException(503, "Model not loaded")

        t0 = time.time()

        if req.stream:
            async def stream_gen():
                for chunk in engine.stream(
                    req.prompt,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                    context=req.context,
                ):
                    yield chunk

            return StreamingResponse(stream_gen(), media_type="text/plain")

        text = engine.generate(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            context=req.context,
        )
        elapsed = (time.time() - t0) * 1000
        tokens_used = len(engine.tok.encode(req.prompt + text))
        return GenerateResponse(
            text=text,
            tokens_used=tokens_used,
            elapsed_ms=round(elapsed, 2),
        )

    # ── Embed ─────────────────────────────────────────────────────────────────
    @app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(_verify_api_key)])
    async def embed_endpoint(req: EmbedRequest, request: Request):
        _check_rate_limit(request)
        if engine is None:
            raise HTTPException(503, "Model not loaded")
        emb = engine.embed(req.texts, pooling=req.pooling)
        return EmbedResponse(
            embeddings=emb.tolist(),
            dim=emb.shape[-1],
        )

    # ── RAG Ingest ────────────────────────────────────────────────────────────
    @app.post("/rag/ingest", dependencies=[Depends(_verify_api_key)])
    async def rag_ingest_endpoint(req: RAGIngestRequest, request: Request):
        _check_rate_limit(request)
        if rag_pipeline is None:
            raise HTTPException(503, "RAG not enabled")
        n_chunks = rag_pipeline.ingest_documents(req.texts)
        return {"status": "ok", "chunks_ingested": n_chunks}

    # ── RAG Query ─────────────────────────────────────────────────────────────
    @app.post("/rag/query", response_model=RAGQueryResponse, dependencies=[Depends(_verify_api_key)])
    async def rag_query_endpoint(req: RAGQueryRequest, request: Request):
        _check_rate_limit(request)
        if rag_pipeline is None:
            raise HTTPException(503, "RAG not enabled")
        if engine is None:
            raise HTTPException(503, "Model not loaded")

        t0 = time.time()
        context = rag_pipeline.get_context(req.query, k=req.top_k_docs)
        answer = engine.generate(
            req.query,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            context=context if context else None,
        )
        elapsed = (time.time() - t0) * 1000
        return RAGQueryResponse(
            answer=answer,
            context_used=context[:500] if context else "",
            elapsed_ms=round(elapsed, 2),
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    logger.info(
        "Starting API server on %s:%d", config.api_host, config.api_port
    )
    uvicorn.run(app, host=config.api_host, port=config.api_port)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLM API Server")
    parser.add_argument("--config", default="config.yaml", help="config file")
    parser.add_argument(
        "--checkpoint", required=True, help="model checkpoint path"
    )
    args = parser.parse_args()

    from config import Config

    cfg = (
        Config.from_yaml(args.config)
        if Path(args.config).exists()
        else Config()
    )
    run_api(cfg, args.checkpoint)
