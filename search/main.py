import logging
import os
import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

# FORCE OFFLINE MODE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FASTEMBED_OFFLINE"] = "1"

import httpx
from fastembed import SparseTextEmbedding
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

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

# OPTION V9: Final Balanced (Personalization + Optimal Limits)
DENSE_PREFETCH_K = 100
SPARSE_PREFETCH_K = 100
RETRIEVE_K = 60
RERANK_LIMIT = 25 
MAX_CHARS = 5000 

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")

# --- Authentication Logic from WORKING V7 ---
def get_upstream_request_kwargs() -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}
    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
    elif API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return kwargs

class SearchAPIRequest(BaseModel):
    question: Any

class SearchAPIItem(BaseModel):
    message_ids: list[str]

class SearchAPIResponse(BaseModel):
    results: list[SearchAPIItem]

class DenseEmbeddingResponse(BaseModel):
    data: list[dict[str, Any]]

class SparseVector(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)

@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name="Qdrant/bm25")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=60.0)
    app.state.qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=API_KEY)
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()

app = FastAPI(title="Search Service V8-Fixed", version="1.2.1", lifespan=lifespan)

async def embed_dense(client: httpx.AsyncClient, text: str) -> list[float]:
    kwargs = get_upstream_request_kwargs()
    for attempt in range(3):
        try:
            response = await client.post(
                EMBEDDINGS_DENSE_URL,
                **kwargs,
                json={"model": EMBEDDINGS_DENSE_MODEL, "input": [text[:MAX_CHARS]]},
                timeout=120.0
            )
            response.raise_for_status()
            res_json = response.json()
            if "data" in res_json:
                return res_json["data"][0]["embedding"]
            return res_json["embedding"]
        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout):
            if attempt == 2: raise
            await asyncio.sleep(2)
    return []

async def embed_sparse(text: str) -> SparseVector:
    vectors = list(get_sparse_model().embed([text[:MAX_CHARS]]))
    item = vectors[0]
    return SparseVector(indices=item.indices.tolist(), values=item.values.tolist())

async def get_rerank_scores(client: httpx.AsyncClient, query: str, targets: list[str]) -> list[float]:
    if not targets: return []
    kwargs = get_upstream_request_kwargs()
    for attempt in range(3):
        try:
            response = await client.post(
                RERANKER_URL,
                **kwargs,
                json={"model": RERANKER_MODEL, "text_1": query[:MAX_CHARS], "text_2": [t[:MAX_CHARS] for t in targets]},
            )
            if response.status_code == 429:
                await asyncio.sleep((attempt + 1) * 5)
                continue
            response.raise_for_status()
            res_json = response.json()
            if "data" in res_json:
                return [float(s["score"]) for s in res_json["data"]]
            return [float(s["score"]) for s in res_json]
        except:
            if attempt == 2: raise
            await asyncio.sleep(1)
    return []

@app.get("/health")
async def health(): return {"status": "ok"}

@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest):
    q_data = payload.question if isinstance(payload.question, dict) else payload.question.dict()
    query = q_data.get("text", "").strip()
    asker = q_data.get("asker", "").strip()
    
    if not query: raise HTTPException(status_code=400, detail="Text required")

    # Персонализация: добавляем автора в запрос для BM25
    sparse_query = query
    if asker:
        sparse_query += f" {asker}"

    client: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant

    dense_vec = await embed_dense(client, query)
    sparse_vec = await embed_sparse(sparse_query)
    
    response = await qdrant.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_vec, using=QDRANT_DENSE_VECTOR_NAME, limit=DENSE_PREFETCH_K),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                using=QDRANT_SPARSE_VECTOR_NAME,
                limit=SPARSE_PREFETCH_K,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=RETRIEVE_K,
        with_payload=True,
    )

    if not response.points:
        return SearchAPIResponse(results=[])

    points = response.points
    rerank_candidates = points[:RERANK_LIMIT]
    rerank_targets = [p.payload.get("page_content", "") for p in rerank_candidates]
    scores = await get_rerank_scores(client, query, rerank_targets)

    # Use len(scores) to avoid zip mismatch crash
    scored_points = []
    date_hints = q_data.get("date_mentions", [])
    for i, p in enumerate(rerank_candidates[:len(scores)]):
        s = scores[i]
        page_content = p.payload.get("page_content", "")
        # 1. Author Boost (+10%)
        if asker and f"author: {asker}" in page_content:
            s *= 1.1
        # 2. Date Boost (+15%)
        if date_hints:
            for hint in date_hints:
                if hint in page_content:
                    s *= 1.15
                    break
        scored_points.append((s, p))

    scored_points = sorted(scored_points, key=lambda x: x[0], reverse=True)
    
    final_message_ids = []
    seen_ids = set()
    
    for _, point in scored_points:
        meta = point.payload.get("metadata") or point.payload
        for mid in meta.get("message_ids", []):
            smid = str(mid)
            if smid not in seen_ids:
                final_message_ids.append(smid)
                seen_ids.add(smid)
        if len(final_message_ids) >= 50: break

    if len(final_message_ids) < 50:
        for point in points[RERANK_LIMIT:]:
            meta = point.payload.get("metadata") or point.payload
            for mid in meta.get("message_ids", []):
                smid = str(mid)
                if smid not in seen_ids:
                    final_message_ids.append(smid)
                    seen_ids.add(smid)
            if len(final_message_ids) >= 50: break

    return SearchAPIResponse(results=[SearchAPIItem(message_ids=final_message_ids[:50])])

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
