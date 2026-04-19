import logging
import os
import asyncio
import json
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

# V21 SUPER-ENSEMBLE LIMITS (Focused on Recall)
DENSE_PREFETCH_K = 100
SPARSE_PREFETCH_K = 100
RETRIEVE_K = 60
RERANK_LIMIT = 25 
MAX_CHARS = 5000 

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")

def get_upstream_request_kwargs() -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}
    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
    elif API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return kwargs

class SearchAPIItem(BaseModel):
    message_ids: list[str]

class SearchAPIResponse(BaseModel):
    results: list[SearchAPIItem]

@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name="Qdrant/bm25")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=120.0)
    app.state.qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=API_KEY)
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()

app = FastAPI(title="Search Service V21 Super-Ensemble", version="7.0.0", lifespan=lifespan)

async def embed_dense(client: httpx.AsyncClient, text: str) -> list[float]:
    kwargs = get_upstream_request_kwargs()
    for attempt in range(3):
        try:
            response = await client.post(
                EMBEDDINGS_DENSE_URL, **kwargs,
                json={"model": EMBEDDINGS_DENSE_MODEL, "input": [text[:MAX_CHARS]]},
                timeout=120.0
            )
            response.raise_for_status()
            res_json = response.json()
            return res_json["data"][0]["embedding"] if "data" in res_json else res_json["embedding"]
        except:
            if attempt == 2: raise
            await asyncio.sleep(2)
    return []

async def embed_sparse(text: str):
    vectors = list(get_sparse_model().embed([text[:MAX_CHARS]]))
    item = vectors[0]
    return {"indices": item.indices.tolist(), "values": item.values.tolist()}

async def get_rerank_scores(client: httpx.AsyncClient, query: str, targets: list[str]) -> list[float]:
    if not targets: return []
    kwargs = get_upstream_request_kwargs()
    for attempt in range(3):
        try:
            response = await client.post(
                RERANKER_URL, **kwargs,
                json={"model": RERANKER_MODEL, "text_1": query[:MAX_CHARS], "text_2": targets},
                timeout=60.0
            )
            if response.status_code == 429:
                await asyncio.sleep((attempt + 1) * 3)
                continue
            response.raise_for_status()
            res_json = response.json()
            return [float(s["score"]) for s in (res_json["data"] if "data" in res_json else res_json)]
        except:
            if attempt == 2: raise
            await asyncio.sleep(1)
    return []

@app.get("/health")
async def health(): return {"status": "ok"}

@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: dict):
    q_data = payload.get("question", {})
    query = q_data.get("text", "").strip()
    search_text = q_data.get("search_text", "").strip() or query
    hyde = q_data.get("hyde", [])
    asker = q_data.get("asker", "").strip()
    
    if not query: raise HTTPException(status_code=400, detail="Text required")

    client: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant

    # V21 SUPER-ENSEMBLE RETRIEVAL (4 Paths)
    # Path 1: Pure Dense
    dense_vec_query = await embed_dense(client, query)
    
    # Path 2: HyDE Dense (Hypothetical Answer)
    dense_vec_hyde = []
    if hyde:
        dense_vec_hyde = await embed_dense(client, hyde[0])

    # Path 3: Personalized Sparse
    sparse_data = await embed_sparse(f"{query} {asker}")
    
    # Path 4: Optimized search_text Sparse
    sparse_optimized = await embed_sparse(search_text)

    # Build Ensemble Prefetch
    prefetches = [
        models.Prefetch(query=dense_vec_query, using=QDRANT_DENSE_VECTOR_NAME, limit=DENSE_PREFETCH_K),
        models.Prefetch(
            query=models.SparseVector(indices=sparse_data["indices"], values=sparse_data["values"]),
            using=QDRANT_SPARSE_VECTOR_NAME,
            limit=SPARSE_PREFETCH_K,
        ),
        models.Prefetch(
            query=models.SparseVector(indices=sparse_optimized["indices"], values=sparse_optimized["values"]),
            using=QDRANT_SPARSE_VECTOR_NAME,
            limit=SPARSE_PREFETCH_K,
        )
    ]
    # Add HyDE if available
    if dense_vec_hyde:
        prefetches.append(models.Prefetch(query=dense_vec_hyde, using=QDRANT_DENSE_VECTOR_NAME, limit=DENSE_PREFETCH_K))

    response = await qdrant.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=prefetches,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=RETRIEVE_K,
        with_payload=True,
    )

    if not response.points:
        return SearchAPIResponse(results=[SearchAPIItem(message_ids=[])])

    points = response.points
    rerank_candidates = points[:RERANK_LIMIT]
    rerank_targets = [p.payload.get("page_content", "")[:MAX_CHARS] for p in rerank_candidates]
    scores = await get_rerank_scores(client, query, rerank_targets)

    # Final Boosting & Collecting
    scored_points = []
    entities = q_data.get("entities", {})
    docs = entities.get("documents", [])
    date_hints = q_data.get("date_mentions", [])

    for i, p in enumerate(rerank_candidates[:len(scores)]):
        s = scores[i]
        page_content = p.payload.get("page_content", "")
        if docs:
            for doc in docs:
                if doc.lower() in page_content.lower(): s *= 1.2; break
        if date_hints:
            for hint in date_hints:
                if hint in page_content: s *= 1.15; break
        if asker and f"author: {asker}" in page_content: s *= 1.1
        scored_points.append((s, p))

    scored_points = sorted(scored_points, key=lambda x: x[0], reverse=True)
    
    final_message_ids = []
    seen_ids = set()
    
    # PASS 1: Diverse Leaders
    for _, point in scored_points[:10]:
        meta = point.payload.get("metadata") or point.payload
        m_ids = meta.get("message_ids", [])
        for mid in m_ids[:2]:
            smid = str(mid)
            if smid not in seen_ids:
                final_message_ids.append(smid); seen_ids.add(smid)

    # PASS 2: Recall Fill
    all_points = [p for _, p in scored_points] + points[len(scored_points):]
    for p in all_points:
        meta = p.payload.get("metadata") or p.payload
        m_ids = meta.get("message_ids", [])
        for mid in m_ids:
            smid = str(mid)
            if smid not in seen_ids:
                final_message_ids.append(smid); seen_ids.add(smid)
        if len(final_message_ids) >= 50: break

    return SearchAPIResponse(results=[SearchAPIItem(message_ids=final_message_ids[:50])])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
