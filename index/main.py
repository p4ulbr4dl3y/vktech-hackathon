import asyncio
import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Tuple

# Режим работы OFFLINE для соответствия требованиям безопасности VK Tech
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FASTEMBED_OFFLINE"] = "1"

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Настройка логирования для отслеживания процесса индексации
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")


class Message(BaseModel):
    """Модель сообщения из входного потока данных."""
    id: str
    thread_sn: str | None = None
    time: int
    text: str
    sender_id: str
    file_snippets: str | list[dict[str, Any]] = ""
    parts: list[dict[str, Any]] | None = None
    mentions: list[str] | None = None


class IndexAPIItem(BaseModel):
    """Модель выходного чанка для сохранения в Qdrant."""
    page_content: str
    dense_content: str
    sparse_content: str
    message_ids: list[str]


class IndexAPIResponse(BaseModel):
    """Модель ответа сервиса индексации."""
    results: list[IndexAPIItem]


app = FastAPI(title="VK Search Indexer", version="5.1.0")

# Настройки сегментации текста
CHUNK_SIZE = 512       # Целевой размер чанка в символах
OVERLAP_SIZE = 128     # Размер перекрытия между чанками
MAX_CHARS = 5000       # Максимальная длина текста для API эмбеддингов
UVICORN_WORKERS = 8    # Количество воркеров сервера


def render_v20(m: Message, chat_name: str, chat_type: str) -> str:
    """
    Формирует обогащенное текстовое представление сообщения.
    Добавляет метаданные (автор, чат, дата) как семантические якоря.
    """
    parts = []
    # Контекст чата
    parts.append(f"chat: {chat_name} type: {chat_type}")

    # Идентификаторы автора и треда
    parts.append(f"author: {m.sender_id}")
    if m.thread_sn:
        parts.append(f"thread: {m.thread_sn}")
    if m.time:
        dt = datetime.fromtimestamp(m.time).strftime("%Y-%m-%d")
        parts.append(f"date: {dt}")

    # Основной текст
    if m.text:
        parts.append(m.text)
    if m.parts:
        for p in m.parts:
            t = p.get("text")
            if t:
                parts.append(t)

    # Информация о файлах и упоминаниях
    if m.mentions:
        parts.append(f"mentions: {' '.join([str(x) for x in m.mentions])}")
    if m.file_snippets:
        try:
            fs = m.file_snippets
            snippets = json.loads(fs) if isinstance(fs, str) else fs
            for s in snippets:
                if isinstance(s, dict) and s.get("name"):
                    parts.append(f"файл: {s['name']}")
        except Exception:
            pass

    return "\n".join(parts).strip()


def build_chunks(
    overlap_messages: list[Message], new_messages: list[Message], chat_info: dict
) -> list[IndexAPIItem]:
    """
    Выполняет символьную нарезку текста на чанки с сохранением оверлапа.
    Обеспечивает связь чанков с идентификаторами сообщений.
    """
    chat_name = chat_info.get("name", "Unknown")
    chat_type = chat_info.get("type", "Unknown")

    def get_text_data(messages: List[Message]) -> Tuple[str, List[Tuple[int, int, str]]]:
        """Собирает полный текст из списка сообщений и фиксирует границы каждого ID."""
        full_text = ""
        ranges = []
        curr = 0
        for m in messages:
            txt = render_v20(m, chat_name, chat_type)
            if not txt:
                continue
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

    if not new_text:
        return []

    # Определяем точку начала для индексации новых сообщений
    start_offset = len(all_text) - len(new_text)
    results = []

    # Генерация чанков методом скользящего окна
    for start in range(start_offset, len(all_text), CHUNK_SIZE - OVERLAP_SIZE):
        end = start + CHUNK_SIZE
        chunk_txt = all_text[start:end]
        if not chunk_txt:
            continue

        # Сопоставление текста чанка с ID сообщений, которые в него попали
        chunk_ids = [mid for s, e, mid in all_ranges if e > start and s < end]
        if not chunk_ids:
            continue

        results.append(
            IndexAPIItem(
                page_content=chunk_txt,
                dense_content=chunk_txt[:MAX_CHARS],
                sparse_content=chunk_txt[:MAX_CHARS],
                message_ids=list(dict.fromkeys(chunk_ids)),
            )
        )
        if end >= len(all_text):
            break
    return results


@app.get("/health")
async def health() -> dict[str, str]:
    """Проверка жизнеспособности сервиса."""
    return {"status": "ok"}


@app.post("/index", response_model=IndexAPIResponse)
async def index(payload: dict) -> IndexAPIResponse:
    """Точка входа для индексации новой порции сообщений."""
    data = payload.get("data", {})
    chat = data.get("chat", {})
    overlap = [Message(**m) for m in data.get("overlap_messages", [])]
    new_msgs = [Message(**m) for m in data.get("new_messages", [])]
    return IndexAPIResponse(results=build_chunks(overlap, new_msgs, chat))


@lru_cache(maxsize=1)
def get_sparse_model() -> Any:
    """Инициализация модели BM25 с кешированием в памяти."""
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name="Qdrant/bm25")


@app.post("/sparse_embedding")
async def sparse_embedding(payload: dict) -> dict[str, Any]:
    """Генерация разреженных векторов для текстовых данных."""
    model = get_sparse_model()
    vectors = []
    for item in model.embed(payload.get("texts", [])):
        vectors.append({"indices": item.indices.tolist(), "values": item.values.tolist()})
    return {"vectors": vectors}


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, workers=UVICORN_WORKERS)
