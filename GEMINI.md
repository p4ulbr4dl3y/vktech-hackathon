# Gemini Context: VK Workspace Message Search (Hackathon Solution)

This repository contains a record-breaking RAG (Retrieval-Augmented Generation) solution for the VK Tech Hackathon 2026. The system is designed for high-precision search across fragmented messenger data in an air-gapped corporate environment.

## 🏗 System Architecture

The solution implements a two-stage retrieval pipeline with a "Super-Ensemble" of search strategies:

1.  **Indexing Service (`index/`)**:
    *   **Enrichment**: Messages are rendered into structured text with semantic anchors (`chat:`, `author:`, `date:`, `file:`, `thread:`).
    *   **Chunking**: Uses a sliding window (default: 512 chars with 128 overlap) to maintain conversational context.
    *   **Sparse Vectors**: Generates BM25 embeddings locally using `fastembed`.
    *   **Security**: Operates in `OFFLINE` mode to satisfy air-gap requirements.

2.  **Search Service (`search/`)**:
    *   **Alpha-Blending Ensemble**: Parallel retrieval using Dense (original query + HyDE) and Sparse (query + optimized search text) vectors.
    *   **Fusion**: Results are combined using **RRF (Reciprocal Rank Fusion)** in Qdrant.
    *   **Cross-Encoder Reranking**: The top 35 candidates are reranked using the `Llama-Nemotron-Reranker-1B` model.
    *   **Heuristic Boosting**: Final scores are boosted based on entity matching (documents, dates) and author relevance.
    *   **Diversification**: Selection logic ensures the top 50 results represent the most relevant messages without redundant overlap.

3.  **Inference & Storage**:
    *   **Qdrant**: Vector database for dense and sparse retrieval.
    *   **External APIs**: Dense embeddings (`Qwen3-Embedding`) and Reranking are provided via external HTTP endpoints during evaluation.

## 🚀 Key Commands

### Deployment
- **Start Services**: `docker-compose up --build`
  - Index Service: `http://localhost:8001`
  - Search Service: `http://localhost:8002`
  - Qdrant: `http://localhost:6333`

### Development & Testing
- **Generate Synthetic Data**: `python generate_stress_data.py`
- **Verify Data**: `python verify_stress_data.py`
- **Full Evaluation**: `python full_evaluator_v3.py` (Calculates Recall@50 and nDCG@50).

## 🛠 Technology Stack
- **Framework**: FastAPI (Python 3.12+)
- **Vector DB**: Qdrant 1.14.1
- **NLP/Embeddings**: `fastembed` (BM25), `httpx` (Async API calls)
- **Validation**: Pydantic v2
- **Server**: Uvicorn (multi-worker configuration)

## 📝 Development Conventions
- **Contract Integrity**: Strictly adhere to schemas defined in `SPECIFICATION.md`. Never change `/index`, `/search`, or `/sparse_embedding` signatures.
- **Offline First**: All models and libraries MUST work without internet access. Ensure `HF_HUB_OFFLINE=1` is respected.
- **Performance**: Target latency is < 60s per query. The "Masterpiece" configuration uses `DENSE_LIMIT=150` and `RERANK_LIMIT=35` as optimal performance/quality trade-offs.
- **Logging**: Use `LOG_LEVEL=INFO` for production-grade tracing of the retrieval pipeline.

## 📂 Project Structure
- `index/`: Indexing microservice.
- `search/`: Search microservice (retrieval & reranking logic).
- `data/`: Sample datasets and ground-truth questions.
- `full_evaluator_v3.py`: The primary metric validation script.
- `SPECIFICATION.md`: Detailed technical requirements and API documentation.
- `CHRONOLOGY.md`: History of score improvements (from 0.46 to 0.6014).
