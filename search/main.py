import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncGenerator

# Режим OFFLINE для обеспечения сетевой изоляции по ТЗ
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FASTEMBED_OFFLINE"] = "1"

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastembed import SparseTextEmbedding
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

# Идентификаторы моделей
EMBEDDINGS_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"

# Конфигурация из переменных окружения
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
API_KEY = os.getenv("API_KEY")
EMBEDDINGS_DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL", "")
RERANKER_URL = os.getenv("RERANKER_URL", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
OPEN_API_LOGIN = os.getenv("OPEN_API_LOGIN")
OPEN_API_PASSWORD = os.getenv("OPEN_API_PASSWORD")

# V23 ALPHA-BLENDING LIMITS: Приоритизация семантики через глубину поиска
DENSE_LIMIT = 150
HYDE_LIMIT = 120
SPARSE_OPT_LIMIT = 100
SPARSE_MAIN_LIMIT = 60
RETRIEVE_K = 80
RERANK_LIMIT = 35
MAX_CHARS = 5000

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")


def get_auth_kwargs() -> dict[str, Any]:
    """Формирование параметров авторизации."""
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}
    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
    elif API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return kwargs


class SearchAPIItem(BaseModel):
    """Модель элемента поисковой выдачи."""

    message_ids: list[str]


class SearchAPIResponse(BaseModel):
    """Модель ответа поискового сервиса."""

    results: list[SearchAPIItem]


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    """Инициализация модели разреженных эмбеддингов BM25."""
    return SparseTextEmbedding(model_name="Qdrant/bm25")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Управление жизненным циклом приложения: инициализация и закрытие клиентов."""
    app.state.http = httpx.AsyncClient(timeout=120.0)
    app.state.qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=API_KEY)
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()


app = FastAPI(title="VK Search Engine V23", version="9.0.0", lifespan=lifespan)


async def embed_dense(client: httpx.AsyncClient, text: str) -> list[float]:
    """Получение плотных эмбеддингов через внешний API с механизмом повторов."""
    kwargs = get_auth_kwargs()
    for attempt in range(5):
        try:
            response = await client.post(
                EMBEDDINGS_DENSE_URL,
                **kwargs,
                json={"model": EMBEDDINGS_DENSE_MODEL, "input": [text[:MAX_CHARS]]},
                timeout=120.0,
            )
            if response.status_code == 502:
                logger.warning(f"Embedding API 502 (attempt {attempt+1}). Retrying...")
                await asyncio.sleep((2**attempt) + 1)
                continue
            response.raise_for_status()
            res_json = response.json()
            if "data" in res_json:
                return res_json["data"][0]["embedding"]
            return res_json["embedding"]
        except Exception as e:
            if attempt == 4:
                raise
            delay = (2**attempt) + 1
            logger.warning(f"Embedding API failed ({e}). Retrying in {delay}s...")
            await asyncio.sleep(delay)
    return []


async def embed_sparse(text: str) -> dict[str, Any]:
    """Локальная генерация разреженных векторов (BM25)."""
    vectors = list(get_sparse_model().embed([text[:MAX_CHARS]]))
    item = vectors[0]
    return {"indices": item.indices.tolist(), "values": item.values.tolist()}


async def get_rerank_scores(
    client: httpx.AsyncClient, query: str, targets: list[str]
) -> list[float]:
    """Получение оценок релевантности от кросс-энкодера."""
    if not targets:
        return []
    kwargs = get_auth_kwargs()
    for attempt in range(3):
        try:
            response = await client.post(
                RERANKER_URL,
                **kwargs,
                json={
                    "model": RERANKER_MODEL,
                    "text_1": query[:MAX_CHARS],
                    "text_2": targets,
                },
                timeout=60.0,
            )
            if response.status_code == 429:
                await asyncio.sleep((attempt + 1) * 3)
                continue
            response.raise_for_status()
            res_json = response.json()
            data = res_json["data"] if "data" in res_json else res_json
            return [float(s["score"]) for s in data]
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(1)
    return []


@app.get("/health")
async def health() -> dict[str, str]:
    """Проверка доступности сервиса."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: dict) -> SearchAPIResponse:
    """
    Основная точка входа для поиска.
    Реализует многостадийный пайплайн: гибридный поиск, RRF-слияние и реранжирование.
    """
    start_time = time.perf_counter()
    q_data = payload.get("question", {})
    query = q_data.get("text", "").strip()
    search_text = q_data.get("search_text", "").strip() or query
    hyde = q_data.get("hyde", [])
    asker = q_data.get("asker", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Text required")

    client: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant

    # ЭТАП 1: Параллельное извлечение кандидатов с весовыми лимитами
    t_retr_start = time.perf_counter()
    dense_task = embed_dense(client, query)
    hyde_task = embed_dense(client, hyde[0]) if hyde else asyncio.sleep(0, [])
    dense_vec, hyde_vec = await asyncio.gather(dense_task, hyde_task)

    sparse_main = await embed_sparse(f"{query} {asker}")
    sparse_opt = await embed_sparse(search_text)

    prefetches = [
        models.Prefetch(
            query=dense_vec, using=QDRANT_DENSE_VECTOR_NAME, limit=DENSE_LIMIT
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_main),
            using=QDRANT_SPARSE_VECTOR_NAME,
            limit=SPARSE_MAIN_LIMIT,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_opt),
            using=QDRANT_SPARSE_VECTOR_NAME,
            limit=SPARSE_OPT_LIMIT,
        ),
    ]
    if hyde_vec:
        prefetches.append(
            models.Prefetch(
                query=hyde_vec, using=QDRANT_DENSE_VECTOR_NAME, limit=HYDE_LIMIT
            )
        )

    response = await qdrant.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=prefetches,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=RETRIEVE_K,
        with_payload=True,
    )
    t_retr = time.perf_counter() - t_retr_start

    if not response.points:
        return SearchAPIResponse(results=[SearchAPIItem(message_ids=[])])

    # ЭТАП 2: Реранкинг
    t_rank_start = time.perf_counter()
    points = response.points
    candidates = points[:RERANK_LIMIT]
    scores = await get_rerank_scores(
        client,
        query,
        [p.payload.get("page_content", "")[:MAX_CHARS] for p in candidates],
    )

    scored = []
    docs = q_data.get("entities", {}).get("documents", [])
    dates = q_data.get("date_mentions", [])

    for i, p in enumerate(candidates[: len(scores)]):
        s = scores[i]
        text = str(p.payload.get("page_content", "")).lower()
        if docs and any(str(d).lower() in text for d in docs):
            s *= 1.2
        if dates and any(str(d) in text for d in dates):
            s *= 1.15
        if asker and f"author: {asker}" in text:
            s *= 1.1
        scored.append((s, p))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    t_rank = time.perf_counter() - t_rank_start

    # ЭТАП 3: Формирование финальной выдачи
    final_ids: list[str] = []
    seen: set[str] = set()
    for _, p in scored[:10]:
        meta = p.payload.get("metadata") or p.payload
        m_ids = meta.get("message_ids", [])
        for mid in m_ids[:2]:
            smid = str(mid)
            if smid not in seen:
                final_ids.append(smid)
                seen.add(smid)

    all_points_seq = [s[1] for s in scored] + points[len(scored) :]
    for p in all_points_seq:
        meta = p.payload.get("metadata") or p.payload
        m_ids = meta.get("message_ids", [])
        for mid in m_ids:
            smid = str(mid)
            if smid not in seen:
                final_ids.append(smid)
                seen.add(smid)
        if len(final_ids) >= 50:
            break

    latency = time.perf_counter() - start_time
    logger.info(f"Search Finished (V23): latency={latency:.3f}s")
    return SearchAPIResponse(results=[SearchAPIItem(message_ids=final_ids[:50])])


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
