import faiss
import pickle as pkl
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
from app.config import STORAGE
from collections import defaultdict, deque
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
bm25 = None
# session_id → deque of (user, assistant)
chat_memory = defaultdict(lambda: deque(maxlen=6))
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

modell = SentenceTransformer("BAAI/bge-base-en-v1.5")

model = AutoModelForCausalLM.from_pretrained(
    "/models/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
    model_type="llama",
    context_length=2048,
    max_new_tokens=128
)

index = None
chunks = None


def reload_index():
    global index, chunks, bm25

    index_path = f"{STORAGE}/my_faiss.index"
    chunks_path = f"{STORAGE}/chunks.pkl"

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        print("No vector store yet. Waiting for ingestion.")
        index = None
        chunks = None
        return

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pkl.load(f)

    print("Loaded", len(chunks), "chunks")
    tokenized_chunks = [word_tokenize(c["text"].lower()) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

def retrieve_chunks(question, k=10):
    # Semantic
    D, I = index.search(embed_and_normalize_query(question), k)
    semantic_hits = I[0].tolist()

    # Keyword
    tokenized_query = word_tokenize(question.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    keyword_hits = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]

    # Merge
    candidates = list(set(semantic_hits + keyword_hits))

    # Rerank
    pairs = [(question, chunks[i]["text"]) for i in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [i for i,_ in ranked]


def embed_and_normalize_query(query):
    emb = modell.encode(query)
    emb = emb / np.linalg.norm(emb)
    return emb.reshape(1, -1)


def answer_question(question,session_id):
    global index, chunks, bm25

    if index is None:
        reload_index()

    if index is None:
        return "No documents ingested yet."
    history = chat_memory[session_id]
    history_text = ""
    for u, a in history:
        history_text += f"User: {u}\nAssistant: {a}\n"

    # Semantic
    D, I = index.search(embed_and_normalize_query(question), 10)
    semantic_hits = I[0].tolist()

    # Keyword
    tokenized_query = word_tokenize(question.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    keyword_hits = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]

    # Merge
    candidates = list(set(semantic_hits + keyword_hits))
    pairs = [(question, chunks[i]["text"]) for i in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top = [chunks[i]["text"] for i,_ in ranked[:3]]

    candidates_text = [chunks[i]["text"] for i in candidates]
    context = "\n".join(top)

    prompt = f"""
You are a study assistant.
Use ONLY the information in the CONTEXT.
If not found, say "I don't know".
CHAT HISTORY:
{history_text}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    print("Waiting here")
    answer = model(prompt)

    history.append((question, answer))

    return answer

