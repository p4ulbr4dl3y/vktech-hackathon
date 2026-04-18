import logging
import os
import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any
from datetime import datetime

import httpx
from fastembed import SparseTextEmbedding
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

# --- Configuration & Environment ---
EMBEDDINGS_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

API_KEY = os.getenv("API_KEY")
EMBEDDINGS_DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL")
RERANKER_URL = os.getenv("RERANKER_URL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")

OPEN_API_LOGIN = os.getenv("OPEN_API_LOGIN")
OPEN_API_PASSWORD = os.getenv("OPEN_API_PASSWORD")

# Optimized retrieval limits
DENSE_PREFETCH_K = 150
SPARSE_PREFETCH_K = 150
RETRIEVE_K = 100
RERANK_LIMIT = 100

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")

# --- Models ---
class DateRange(BaseModel):
    from_: str = Field(alias="from")
    to: str

class Entities(BaseModel):
    people: list[str] | None = Field(default_factory=list)
    emails: list[str] | None = Field(default_factory=list)
    documents: list[str] | None = Field(default_factory=list)
    names: list[str] | None = Field(default_factory=list)
    links: list[str] | None = Field(default_factory=list)

class Question(BaseModel):
    text: str
    asker: str = ""
    asked_on: str = ""
    variants: list[str] | None = Field(default_factory=list)
    hyde: list[str] | None = Field(default_factory=list)
    keywords: list[str] | None = Field(default_factory=list)
    entities: Entities | None = None
    date_mentions: list[str] | None = Field(default_factory=list)
    date_range: DateRange | None = None
    search_text: str = ""

class SearchAPIRequest(BaseModel):
    question: Question

class SearchAPIItem(BaseModel):
    message_ids: list[str]

class SearchAPIResponse(BaseModel):
    results: list[SearchAPIItem]

class DenseEmbeddingResponse(BaseModel):
    data: list[dict[str, Any]]

class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]

# --- Query Expander Logic ---
class QueryExpander:
    @staticmethod
    def build_dense(q: Question) -> str:
        parts = [q.text]
        if q.search_text and q.search_text != q.text:
            parts.append(q.search_text)
        if q.hyde:
            parts.extend(q.hyde)
        return " ".join(parts)

    @staticmethod
    def build_sparse(q: Question) -> str:
        parts = [q.text]
        # Massive boost for exact keywords and search_text
        s_text = q.search_text or q.text
        parts.extend([s_text] * 10) 
        if q.keywords:
            for kw in q.keywords:
                parts.extend([kw] * 15)
        # Add names with high weight
        if q.entities and q.entities.names:
            for n in q.entities.names:
                parts.extend([n] * 5)
        return " ".join(parts)

    @staticmethod
    def build_rerank(q: Question) -> str:
        parts = [q.text]
        if q.search_text and q.search_text != q.text:
            parts.append(q.search_text)
        if q.keywords:
            parts.extend(q.keywords)
        return " ".join(parts)

    @staticmethod
    def build_filter(q: Question) -> models.Filter | None:
        must = []
        # Entities filter
        if q.entities:
            # Note: payload path might need adjustment based on how Index Service saves it
            # Baseline doesn't have nested entities yet, but our enhanced Index Service might
            if q.entities.people:
                must.append(models.FieldCondition(key="participants", match=models.MatchAny(any=q.entities.people)))
            # We can also add email/link filters if we index them
        
        # Date range
        if q.date_range:
            try:
                def to_ts(s): return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
                # Note: Baseline indexer doesn't save timestamps as ints yet, need to be careful
                # If using baseline metadata format, filtering by date might require Index Service updates
                pass 
            except:
                pass

        return models.Filter(must=must) if must else None

# --- Services ---
def get_upstream_request_kwargs() -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}
    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
    elif API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return kwargs

@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name="Qdrant/bm25")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=30.0)
    app.state.qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=API_KEY)
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()

app = FastAPI(title="Search Service Enhanced", version="1.0.0", lifespan=lifespan)

async def embed_dense(client: httpx.AsyncClient, text: str) -> list[float]:
    resp = await client.post(
        EMBEDDINGS_DENSE_URL,
        **get_upstream_request_kwargs(),
        json={"model": EMBEDDINGS_DENSE_MODEL, "input": [text]}
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]

async def embed_sparse(text: str) -> SparseVector:
    vectors = list(get_sparse_model().embed([text]))
    item = vectors[0]
    return SparseVector(indices=item.indices.tolist(), values=item.values.tolist())

async def get_rerank_scores(client: httpx.AsyncClient, query: str, targets: list[str]) -> list[float]:
    if not targets: return []
    resp = await client.post(
        RERANKER_URL,
        **get_upstream_request_kwargs(),
        json={"model": RERANKER_MODEL, "text_1": query, "text_2": targets}
    )
    resp.raise_for_status()
    return [float(s["score"]) for s in resp.json().get("data", [])]

@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest) -> SearchAPIResponse:
    q = payload.question
    query_text = q.text.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Text required")

    client: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant

    # 1. Expand Queries
    dense_q = QueryExpander.build_dense(q)
    sparse_q = QueryExpander.build_sparse(q)
    rerank_q = QueryExpander.build_rerank(q)
    search_filter = QueryExpander.build_filter(q)

    # 2. Parallel Embedding
    dense_vec_task = embed_dense(client, dense_q)
    sparse_vec_task = embed_sparse(sparse_q)
    dense_vec, sparse_vec = await asyncio.gather(dense_vec_task, sparse_vec_task)

    # 3. Hybrid Search
    search_result = await qdrant.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vec, using=QDRANT_DENSE_VECTOR_NAME, limit=DENSE_PREFETCH_K),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                using=QDRANT_SPARSE_VECTOR_NAME,
                limit=SPARSE_PREFETCH_K
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=search_filter,
        limit=RETRIEVE_K,
        with_payload=True
    )

    if not search_result.points:
        return SearchAPIResponse(results=[SearchAPIItem(message_ids=[])])

    # 4. Rerank Top candidates
    points = search_result.points
    rerank_targets = [p.payload.get("page_content", "") for p in points[:RERANK_LIMIT]]
    scores = await get_rerank_scores(client, rerank_q, rerank_targets)
    
    # Sort by rerank score
    scored_points = sorted(zip(scores, points), key=lambda x: x[0], reverse=True)
    for s, p in scored_points[:5]:
        logger.info(f"Top point score {s:.4f}: {p.payload.get('page_content')[:50]}...")
    
    # 5. Extract Unique Message IDs (keep order)
    # We take IDs from top ranked points
    final_message_ids = []
    seen_ids = set()
    
    # Precision trick: we trust reranker's relative ordering.
    for score, point in scored_points:
        meta = point.payload.get("metadata") or point.payload
        m_ids = meta.get("message_ids", [])
        for mid in m_ids:
            smid = str(mid)
            if smid not in seen_ids:
                final_message_ids.append(smid)
                seen_ids.add(smid)
        
        # If we already have enough high-quality IDs, stop
        if len(final_message_ids) >= 20:
            break

    return SearchAPIResponse(results=[SearchAPIItem(message_ids=final_message_ids[:50])])

@app.get("/health")
async def health(): return {"status": "ok"}

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT)
