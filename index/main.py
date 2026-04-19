import logging
import os
import asyncio
import json
from datetime import datetime
from functools import lru_cache
from typing import Any

# FORCE OFFLINE MODE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FASTEMBED_OFFLINE"] = "1"

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")

class Message(BaseModel):
    id: str
    thread_sn: str | None = None
    time: int
    text: str
    sender_id: str
    file_snippets: str | list[dict[str, Any]] = ""
    parts: list[dict[str, Any]] | None = None
    mentions: list[str] | None = None

class IndexAPIItem(BaseModel):
    page_content: str
    dense_content: str
    sparse_content: str
    message_ids: list[str]

class IndexAPIResponse(BaseModel):
    results: list[IndexAPIItem]

app = FastAPI(title="Index Service V20 Masterpiece", version="5.0.0")

# V20: Use 512-char chunks for speed (like V9) but with contextual tags
CHUNK_SIZE = 512
OVERLAP_SIZE = 128
MAX_CHARS = 5000 
UVICORN_WORKERS = 8

def render_v20(m: Message, chat_name: str, chat_type: str) -> str:
    """Ultimate contextual rendering."""
    parts = []
    # 1. Chat Context (The 'Where')
    parts.append(f"chat: {chat_name} type: {chat_type}")
    
    # 2. Author and Thread (The 'Who/How')
    parts.append(f"author: {m.sender_id}")
    if m.thread_sn:
        parts.append(f"thread: {m.thread_sn}")
    if m.time:
        dt = datetime.fromtimestamp(m.time).strftime('%Y-%m-%d')
        parts.append(f"date: {dt}")
    
    # 3. Content
    if m.text: parts.append(m.text)
    if m.parts:
        for p in m.parts:
            t = p.get("text")
            if t: parts.append(t)
    
    # 4. Files and Mentions
    if m.mentions: parts.append(f"mentions: {' '.join([str(x) for x in m.mentions])}")
    if m.file_snippets:
        try:
            fs = m.file_snippets
            snippets = json.loads(fs) if isinstance(fs, str) else fs
            for s in snippets:
                if isinstance(s, dict) and s.get("name"): parts.append(f"файл: {s['name']}")
        except: pass
        
    return "\n".join(parts).strip()

def build_chunks(overlap_messages: list[Message], new_messages: list[Message], chat_info: dict) -> list[IndexAPIItem]:
    """Character-based chunking with V20 contextual rendering."""
    chat_name = chat_info.get("name", "Unknown")
    chat_type = chat_info.get("type", "Unknown")
    
    def get_text_data(messages):
        full_text = ""
        ranges = []
        curr = 0
        for m in messages:
            txt = render_v20(m, chat_name, chat_type)
            if not txt: continue
            if full_text:
                full_text += "\n\n"
                curr += 2
            start = curr
            full_text += txt
            curr += len(txt)
            ranges.append((start, curr, m.id))
        return full_text, ranges

    all_text, all_ranges = get_text_data(overlap_messages + new_messages)
    new_text, _ = get_text_data(new_messages)
    
    if not new_text: return []
    
    start_offset = len(all_text) - len(new_text)
    results = []
    
    for start in range(start_offset, len(all_text), CHUNK_SIZE - OVERLAP_SIZE):
        end = start + CHUNK_SIZE
        chunk_txt = all_text[start:end]
        if not chunk_txt: continue
        
        chunk_ids = [mid for s, e, mid in all_ranges if e > start and s < end]
        if not chunk_ids: continue
        
        results.append(IndexAPIItem(
            page_content=chunk_txt,
            dense_content=chunk_txt[:MAX_CHARS],
            sparse_content=chunk_txt[:MAX_CHARS],
            message_ids=list(dict.fromkeys(chunk_ids))
        ))
        if end >= len(all_text): break
    return results

@app.get("/health")
async def health(): return {"status": "ok"}

@app.post("/index", response_model=IndexAPIResponse)
async def index(payload: dict):
    data = payload.get("data", {})
    chat = data.get("chat", {})
    overlap = [Message(**m) for m in data.get("overlap_messages", [])]
    new_msgs = [Message(**m) for m in data.get("new_messages", [])]
    return IndexAPIResponse(results=build_chunks(overlap, new_msgs, chat))

@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name="Qdrant/bm25")

@app.post("/sparse_embedding")
async def sparse_embedding(payload: dict):
    model = get_sparse_model()
    vectors = []
    for item in model.embed(payload.get("texts", [])):
        vectors.append({"indices": item.indices.tolist(), "values": item.values.tolist()})
    return {"vectors": vectors}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, workers=UVICORN_WORKERS)
