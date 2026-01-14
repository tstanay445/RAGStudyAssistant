import json
from rag import retrieve_chunks, reload_index

reload_index()

with open("eval.json") as f:
    data = json.load(f)

def recall_at_k(k):
    hits = 0
    for item in data:
        retrieved = retrieve_chunks(item["question"], k=k)
        if any(r in item["relevant_chunk_ids"] for r in retrieved[:k]):
            hits += 1
    return hits / len(data)

for k in [1,3,5,10]:
    print(f"Recall@{k}: {recall_at_k(k):.3f}")
