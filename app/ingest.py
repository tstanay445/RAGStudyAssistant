import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import STORAGE
import regex as re

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
os.makedirs(STORAGE, exist_ok=True)

def run_ingestion(path):

    def read_txt(path):
        files = sorted(os.listdir(path))
        content = []

        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                    content.append(f.read())

        print("Content read successfully")
        return content

    content = read_txt(path)

    def good_text(text):
        text = text.replace("\n\n", "\n")
        splits = text.split("\n")
        cleaned = []

        for s in splits:
            cleaned.append(re.sub(r"\s+", " ", s))

        return "\n".join(cleaned)

    docs = [good_text(c) for c in content]

    def chunk_documents(docs, chunk_size=600, overlap=50):
        chunks = []
        chunk_id = 0

        for doc_id, doc in enumerate(docs):
            paragraphs = [p.strip() for p in doc.split("\n") if p.strip()]
            buffer = ""

            for para in paragraphs:
                if len(para) > chunk_size:
                    if buffer:
                        chunks.append({
                            "text": buffer.strip(),
                            "source_doc": doc_id,
                            "chunk_id": chunk_id
                        })
                        chunk_id += 1
                        buffer = ""

                    start = 0
                    while start < len(para):
                        end = start + chunk_size
                        piece = para[start:end]
                        chunks.append({
                            "text": piece,
                            "source_doc": doc_id,
                            "chunk_id": chunk_id
                        })
                        chunk_id += 1
                        start = end - overlap
                    continue

                if len(buffer) + len(para) <= chunk_size:
                    buffer += para + "\n"
                else:
                    chunks.append({
                        "text": buffer.strip(),
                        "source_doc": doc_id,
                        "chunk_id": chunk_id
                    })
                    chunk_id += 1
                    buffer = buffer[-overlap:] + para + "\n"

            if buffer.strip():
                chunks.append({
                    "text": buffer.strip(),
                    "source_doc": doc_id,
                    "chunk_id": chunk_id
                })
                chunk_id += 1

        return chunks

    chunks = chunk_documents(docs)

    embeddings = model.encode([c["text"] for c in chunks])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # ✅ write FAISS + chunks to STORAGE
    faiss.write_index(index, os.path.join(STORAGE, "my_faiss.index"))

    with open(os.path.join(STORAGE, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"Ingested {len(chunks)} chunks")

    return {
        "num_documents": len(content),
        "num_chunks": len(chunks)
    }