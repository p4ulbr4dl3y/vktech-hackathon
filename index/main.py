import logging
import os
import re
import asyncio
import json

# FORCE OFFLINE MODE for VK environment
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FASTEMBED_OFFLINE"] = "1"

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")

# --- Models from TZ ---
class Chat(BaseModel):
    id: str
    name: str
    sn: str
    type: str
    is_public: bool | None = None
    members_count: int | None = None
    members: list[dict[str, Any]] | None = None

class Message(BaseModel):
    id: str
    thread_sn: str | None = None
    time: int
    text: str
    sender_id: str
    file_snippets: str | list[dict[str, Any]] = ""
    parts: list[dict[str, Any]] | None = None
    mentions: list[str] | None = None
    member_event: dict[str, Any] | None = None
    is_system: bool
    is_hidden: bool
    is_forward: bool
    is_quote: bool

class ChatData(BaseModel):
    chat: Chat
    overlap_messages: list[Message]
    new_messages: list[Message]

class IndexAPIRequest(BaseModel):
    data: ChatData

class IndexAPIItem(BaseModel):
    page_content: str
    dense_content: str
    sparse_content: str
    message_ids: list[str]

class IndexAPIResponse(BaseModel):
    results: list[IndexAPIItem]

class SparseEmbeddingRequest(BaseModel):
    texts: list[str]

class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]

class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]

app = FastAPI(title="Index Service Balanced", version="1.1.0")

# Parameters aligned with baseline for maximum precision
CHUNK_SIZE = 512 
OVERLAP_SIZE = 128
MAX_CHARS = 5000 
SPARSE_MODEL_NAME = "Qdrant/bm25"
UVICORN_WORKERS = 8

def render_message(message: Message) -> str:
    """Rich rendering including text, parts, mentions, and file names."""
    text_parts = []
    
    # 1. Основной текст
    if message.text:
        text_parts.append(message.text)
        
    # 2. Части сообщения (включая цитаты и пересылки)
    if message.parts:
        for part in message.parts:
            p_text = part.get("text")
            if p_text:
                text_parts.append(p_text)
                
    # 3. Упоминания (Mentions) - критично для поиска людей
    if message.mentions:
        # Добавляем меншны как отдельные токены для Sparse поиска
        mentions_str = " ".join([str(m) for m in message.mentions])
        if mentions_str:
            text_parts.append(f"mentions: {mentions_str}")

    # 4. Файлы (file_snippets)
    if message.file_snippets:
        try:
            snippets = []
            if isinstance(message.file_snippets, str) and message.file_snippets.startswith("["):
                snippets = json.loads(message.file_snippets)
            elif isinstance(message.file_snippets, list):
                snippets = message.file_snippets
                
            for s in snippets:
                if isinstance(s, dict) and s.get("name"):
                    text_parts.append(f"файл: {s['name']}")
        except:
            pass

    return "\n".join(text_parts).strip()

def build_chunks(overlap_messages: list[Message], new_messages: list[Message]) -> list[IndexAPIItem]:
    """Strict baseline-aligned character chunking with our rendering."""
    result: list[IndexAPIItem] = []

    def build_text_and_ranges(messages: list[Message]) -> tuple[str, list[tuple[int, int, str]]]:
        text_parts: list[str] = []
        message_ranges: list[tuple[int, int, str]] = []
        position = 0
        for index, message in enumerate(messages):
            text = render_message(message)
            if not text: continue
            if index > 0 and text_parts:
                text_parts.append("\n")
                position += 1
            start = position
            text_parts.append(text)
            position += len(text)
            message_ranges.append((start, position, message.id))
        return "".join(text_parts), message_ranges

    def slice_tail(text: str, tail_size: int) -> str:
        if tail_size <= 0: return ""
        return text[max(0, len(text) - tail_size):]

    overlap_text, _ = build_text_and_ranges(overlap_messages)
    previous_chunk_text = slice_tail(overlap_text, OVERLAP_SIZE)
    new_text, new_message_ranges = build_text_and_ranges(new_messages)

    for start in range(0, len(new_text), CHUNK_SIZE):
        chunk_body = new_text[start : start + CHUNK_SIZE]
        if not chunk_body: continue

        # Correct ID mapping for this specific chunk window
        chunk_body_ids = [
            mid for m_start, m_end, mid in new_message_ranges
            if m_end > start and m_start < start + len(chunk_body)
        ]
        
        chunk_text = previous_chunk_text
        if chunk_text and chunk_body: chunk_text += "\n"
        chunk_text += chunk_body

        result.append(
            IndexAPIItem(
                page_content=chunk_text,
                dense_content=chunk_text[:MAX_CHARS],
                sparse_content=chunk_text[:MAX_CHARS],
                message_ids=list(dict.fromkeys(chunk_body_ids)) # Unique IDs
            )
        )
        previous_chunk_text = slice_tail(chunk_text, OVERLAP_SIZE)

    return result

@app.get("/health")
async def health(): return {"status": "ok"}

@app.post("/index", response_model=IndexAPIResponse)
async def index(payload: IndexAPIRequest) -> IndexAPIResponse:
    return IndexAPIResponse(results=build_chunks(payload.data.overlap_messages, payload.data.new_messages))

@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    model = get_sparse_model()
    return [SparseVector(indices=item.indices.tolist(), values=item.values.tolist()) for item in model.embed(texts)]

@app.post("/sparse_embedding")
async def sparse_embedding(payload: SparseEmbeddingRequest):
    vectors = await asyncio.to_thread(embed_sparse_texts, payload.texts)
    return {"vectors": vectors}

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, workers=UVICORN_WORKERS)
