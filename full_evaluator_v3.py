import asyncio
import json
import logging
import math
import os

import httpx
from qdrant_client import QdrantClient, models

# --- Configuration ---
SEARCH_URL = "http://localhost:8002/search"
INDEX_URL = "http://localhost:8001/index"
SPARSE_EMB_URL = "http://localhost:8001/sparse_embedding"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "evaluation"
EMBEDDINGS_DENSE_URL = "http://83.166.249.64:18001/embeddings"

# Credentials
OPEN_API_LOGIN = os.getenv("OPEN_API_LOGIN", "7d96acc1372d61a1")
OPEN_API_PASSWORD = os.getenv("OPEN_API_PASSWORD", "4e4f121c3cac9e1e36d1e76e09a87029")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("full-evaluator")

GOLDEN_SET = [
    {
        "q": {"text": "Опрос по итогам PHP-года для сообщества"},
        "expected": ["3666666666666666666"],
    },
    {
        "q": {"text": "Когда релиз Go 1.18 и какие фичи?"},
        "expected": ["3888888888888888888"],
    },
    {
        "q": {"text": "Митап Nova TechTalk в конце февраля или марте"},
        "expected": ["4111111111111111110"],
    },
    {"q": {"text": "Кто такая Ника devrel?"}, "expected": ["4222222222222222221"]},
    {
        "q": {"text": "SIGABRT: abort PC=0x191919191"},
        "expected": ["4444444444444444444"],
    },
    {
        "q": {"text": "MacBook Air M1 2020 syscall error"},
        "expected": ["4555555555555555555"],
    },
    {
        "q": {"text": "Хакатон Inner Spark для души"},
        "expected": ["5111111111111111110"],
    },
    {
        "q": {"text": "Где хранить скрипты схем баз данных?"},
        "expected": ["5222222222222222221"],
    },
    {
        "q": {"text": "Использование утилиты goose для миграций"},
        "expected": ["5444444444444444443"],
    },
    {
        "q": {"text": "Сборка проекта с CGO под Linux"},
        "expected": ["5555555555555555555", "5888888888888888888"],
    },
    {
        "q": {"text": "Типизация Kafka-топиков и Confluent Schema Registry"},
        "expected": ["4999999999999999999", "6222222222222222221"],
    },
    {
        "q": {"text": "Интерпретатор Quasigo от Тимура Беляева"},
        "expected": ["4999999999999999999", "6222222222222222221"],
    },
    # --- НОВЫЕ КЕЙСЫ ---
    {
        "q": {
            "text": "Техтолки Rocket Deli про автоматизацию и линтер",
            "keywords": ["Rocket Deli", "линтер"],
        },
        "expected": ["3777777777777777777"],
    },
    {
        "q": {
            "text": "Fuzzing тестирование от Антона Жарова",
            "entities": {"people": ["Антон Жаров"]},
        },
        "expected": ["3888888888888888888"],
    },
    {
        "q": {
            "text": "Страничка в интранете про технологии (распознавание лиц)",
            "keywords": ["интранет", "распознавание"],
        },
        "expected": ["3999999999999999999"],
    },
    {
        "q": {"text": "Посмотреть под strace какой сисколл передали"},
        "expected": ["4666666666666666666"],
    },
    {
        "q": {"text": "Приглашение знакомым PHPшникам пройти опрос"},
        "expected": ["3666666666666666666"],
    },
    {
        "q": {
            "text": "Escape analysis и влияние указателей на стек и кучу",
            "keywords": ["Escape analysis", "стек"],
        },
        "expected": ["3888888888888888888"],
    },
    {
        "q": {"text": "Воркшоп: как написать Terraform-провайдер"},
        "expected": ["4999999999999999999", "6222222222222222221"],
    },
    {
        "q": {"text": "Ссылка на картинку IMG_8471.webp"},
        "expected": ["5999999999999999999"],
    },
]


def calculate_ndcg(found_ids, expected_ids, k=50):
    def dcg(rels):
        return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

    rels = [1 if fid in expected_ids else 0 for fid in found_ids[:k]]
    actual = dcg(rels)
    ideal = dcg(sorted([1] * len(expected_ids), reverse=True))
    return actual / ideal if ideal > 0 else 0.0


async def get_dense_embedding(text, client):
    r = await client.post(
        EMBEDDINGS_DENSE_URL,
        auth=(OPEN_API_LOGIN, OPEN_API_PASSWORD),
        json={"model": "Qwen/Qwen3-Embedding-0.6B", "input": [text]},
        timeout=60.0,
    )
    return r.json()["data"][0]["embedding"]


async def run_full_test():
    data_path = "data/stress_test_500.json"
    with open(data_path, "r") as f:
        chat_data = json.load(f)

    async with httpx.AsyncClient() as client:
        # Indexing (Long timeout for model download)
        logger.info("Indexing (might take time due to model download)...")
        try:
            resp = await client.post(
                INDEX_URL,
                json={
                    "data": {
                        "chat": chat_data["chat"],
                        "overlap_messages": [],
                        "new_messages": chat_data["messages"],
                    }
                },
                timeout=300.0,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return

        chunks = resp.json()["results"]
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except:
            pass
        qdrant.create_collection(
            COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

        points = []
        for i, res in enumerate(chunks):
            dv = await get_dense_embedding(res["dense_content"], client)
            sv_r = await client.post(
                SPARSE_EMB_URL, json={"texts": [res["sparse_content"]]}, timeout=300.0
            )
            sv_d = sv_r.json()["vectors"][0]
            points.append(
                models.PointStruct(
                    id=i,
                    vector={
                        "dense": dv,
                        "sparse": models.SparseVector(
                            indices=sv_d["indices"], values=sv_d["values"]
                        ),
                    },
                    payload={
                        "page_content": res["page_content"],
                        "metadata": {"message_ids": res["message_ids"]},
                    },
                )
            )
        qdrant.upsert(COLLECTION_NAME, points=points)

        # Evaluation
        mrr, recall, ndcg = 0, 0, 0
        logger.info("\nEvaluating (with 120s timeout per query)...")
        for item in GOLDEN_SET:
            try:
                r = await client.post(
                    SEARCH_URL, json={"question": item["q"]}, timeout=120.0
                )
                r.raise_for_status()
                found = (
                    r.json()["results"][0]["message_ids"] if r.json()["results"] else []
                )
            except Exception as e:
                logger.error(f"  Search failed for '{item['q']['text'][:30]}': {e}")
                found = []
            rank = next(
                (i + 1 for i, fid in enumerate(found) if fid in item["expected"]), 0
            )
            mrr += 1.0 / rank if rank > 0 else 0
            recall += 1.0 if rank > 0 and rank <= 50 else 0
            ndcg += calculate_ndcg(found, item["expected"])
            status = "✅" if rank == 1 else ("⚠️" if rank > 0 else "❌")
            logger.info(f"{status} Rank {rank} | Q: {item['q']['text'][:40]}...")

        n = len(GOLDEN_SET)
        print("\n" + "=" * 40)
        print(f"Recall@50: {recall/n:.4f}")
        print(f"MRR:       {mrr/n:.4f}")
        print(f"VK SCORE:  {(recall/n)*0.8 + (ndcg/n)*0.2:.4f}")
        print("=" * 40)


if __name__ == "__main__":
    asyncio.run(run_full_test())
