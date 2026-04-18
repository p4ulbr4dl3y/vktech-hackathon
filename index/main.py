import logging
import os
import re
from functools import lru_cache
from typing import Any
import asyncio

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- Environment Variables ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")

# --- Models ---
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

# --- App ---
app = FastAPI(title="Index Service Enhanced", version="1.0.0")

# --- Optimized Parameters ---
CHUNK_SIZE = 3 # Small chunks = High Precision
CHUNK_OVERLAP = 1
SPARSE_MODEL_NAME = "Qdrant/bm25"
FASTEMBED_CACHE_PATH = "/models/fastembed"
UVICORN_WORKERS = 8

def render_message(message: Message) -> str:
    """Rich rendering of message including parts and snippets."""
    text_parts = []
    
    if message.text:
        text_parts.append(message.text)
        
    if message.parts:
        for part in message.parts:
            p_text = part.get("text")
            if p_text:
                text_parts.append(p_text)
                
    # Extract info from file snippets if they are JSON strings
    if isinstance(message.file_snippets, str) and message.file_snippets.startswith("["):
        try:
            import json
            snippets = json.loads(message.file_snippets)
            for s in snippets:
                if s.get("name"):
                    text_parts.append(f"Файл: {s['name']}")
        except:
            pass

    return "\n".join(text_parts).strip()

def build_chunks_enhanced(
    chat: Chat,
    overlap_messages: list[Message],
    new_messages: list[Message],
) -> list[IndexAPIItem]:
    """Sliding window chunking with metadata focus."""
    all_msgs = overlap_messages + new_messages
    # Filter hidden
    active_msgs = [m for m in all_msgs if not m.is_hidden]
    if not active_msgs:
        return []

    results = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    for i in range(0, len(active_msgs), step):
        chunk_msgs = active_msgs[i : i + CHUNK_SIZE]
        if not chunk_msgs:
            break
            
        # Only index if at least one message is from 'new_messages' 
        # (to avoid duplicating work from previous windows, 
        # though the system usually handles this)
        new_ids = {m.id for m in new_messages}
        if not any(m.id in new_ids for m in chunk_msgs):
            continue

        rendered_texts = [render_message(m) for m in chunk_msgs if render_message(m)]
        if not rendered_texts:
            continue

        page_content = "\n---\n".join(rendered_texts)
        
        # Meta-boosting for Sparse: add keywords and mentions
        mentions = []
        for m in chunk_msgs:
            if m.mentions:
                mentions.extend(m.mentions)
        
        sparse_extra = " ".join(list(set(mentions)))
        sparse_content = f"{page_content}\n{sparse_extra}"

        results.append(
            IndexAPIItem(
                page_content=page_content,
                dense_content=page_content,
                sparse_content=sparse_content,
                message_ids=[m.id for m in chunk_msgs]
            )
        )
    return results

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/index", response_model=IndexAPIResponse)
async def index(payload: IndexAPIRequest) -> IndexAPIResponse:
    chunks = build_chunks_enhanced(
        payload.data.chat,
        payload.data.overlap_messages,
        payload.data.new_messages
    )
    return IndexAPIResponse(results=chunks)

@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    model = get_sparse_model()
    vectors = []
    for item in model.embed(texts):
        vectors.append(
            SparseVector(
                indices=item.indices.tolist(),
                values=item.values.tolist()
            )
        )
    return vectors

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
