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

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def run_api(config, checkpoint_path: str) -> None:
    """Start the FastAPI server."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
        import uvicorn
    except ImportError:
        print(
            "FastAPI/uvicorn not installed.\n"
            "Run: pip install fastapi uvicorn"
        )
        sys.exit(1)

    app = FastAPI(
        title="SLM API",
        version="1.0",
        description="Small Language Model — Production API",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine = None
    rag_pipeline = None

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

    # ── Startup ───────────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup():
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

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        return {
            "status": "ok" if engine else "no_model",
            "rag_enabled": rag_pipeline is not None,
        }

    # ── Generate ──────────────────────────────────────────────────────────────
    @app.post("/generate")
    async def generate_endpoint(req: GenerateRequest):
        if engine is None:
            raise HTTPException(503, "Model not loaded")

        t0 = time.time()

        if req.stream:
            async def stream_gen():
                for chunk in engine.stream(
                    req.prompt,
                    req.max_new_tokens,
                    req.temperature,
                    req.top_k,
                    req.top_p,
                    req.context,
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
    @app.post("/embed", response_model=EmbedResponse)
    async def embed_endpoint(req: EmbedRequest):
        if engine is None:
            raise HTTPException(503, "Model not loaded")
        emb = engine.embed(req.texts, pooling=req.pooling)
        return EmbedResponse(
            embeddings=emb.tolist(),
            dim=emb.shape[-1],
        )

    # ── RAG Ingest ────────────────────────────────────────────────────────────
    @app.post("/rag/ingest")
    async def rag_ingest_endpoint(req: RAGIngestRequest):
        if rag_pipeline is None:
            raise HTTPException(503, "RAG not enabled")
        n_chunks = rag_pipeline.ingest_documents(req.texts)
        return {"status": "ok", "chunks_ingested": n_chunks}

    # ── RAG Query ─────────────────────────────────────────────────────────────
    @app.post("/rag/query", response_model=RAGQueryResponse)
    async def rag_query_endpoint(req: RAGQueryRequest):
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
