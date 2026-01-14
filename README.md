# 📚 Production-Grade RAG Study Assistant

A local, production-style Retrieval-Augmented Generation (RAG) system designed for studying technical content.  
The project emphasizes **retrieval quality, grounding, and evaluation**, not just LLM responses.

---

## 🔹 Features

- Paragraph-aware + fallback chunking
- Dense retrieval using **BAAI/bge-base-en-v1.5**
- FAISS vector index with normalized embeddings
- Hybrid retrieval (Dense + BM25)
- Cross-encoder reranking (**ms-marco-MiniLM-L-6-v2**)
- Grounded prompting (hallucination-resistant by design)
- Session-based multi-turn chat memory
- Local LLM inference via GGUF (**ctransformers**)
- FastAPI backend
- Gradio UI
- Offline retrieval evaluation (Recall@K)

---

## 🧠 Architecture

**Pipeline:**

User Question  
↓  
Dense Retrieval (FAISS) + Keyword Retrieval (BM25)  
↓  
Candidate Merge  
↓  
Cross-Encoder Reranking  
↓  
Top-K Context Selection  
↓  
Grounded Prompt → Local LLM



---

## 📊 Evaluation

Retrieval quality is evaluated **offline**, independent of the LLM, using Recall@K.

### Retrieval Metrics (Internal Evaluation)

| Metric   | Value |
|---------|-------|
| Recall@1 | 0.78 |
| Recall@3 | 1.00 |
| Recall@5 | 1.00 |
| Recall@10 | 1.00 |

**Notes:**
- Metrics are computed on a small, manually curated, synthetic evaluation set.
- The goal is to validate retriever correctness, not to claim benchmark performance.
- Evaluation is intentionally decoupled from LLM generation.

Evaluation script:
```bash
python evaluate.py
```

## 🏃 Running Locally
```bash
docker compose up --build
```
Then open:

- API docs: http://localhost:8000/docs
- UI: http://localhost:7860

## 📁 Project Structure
```bash
.
├── app/
│   ├── api.py
│   ├── ingest.py
│   ├── rag.py
│   └── config.py
├── ui.py
├── evaluate.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```
## ⚠️ Important Notes
LLM model files (GGUF) are not included in this repository

Vector stores are generated at runtime

Evaluation datasets are excluded for safety

Designed for local inference and experimentation

## 🚀 Future Improvements
Streaming responses

Batched retrieval and reranking

Citation tracking per answer

Authentication and rate limiting

Optional GPU support

## 📌 Why This Project
Most RAG demos focus only on generation.
This project focuses on retrieval correctness, grounding, and evaluation, reflecting real-world production RAG systems.

