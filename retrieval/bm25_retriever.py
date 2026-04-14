import json
from pathlib import Path
from rank_bm25 import BM25Okapi

PROCESSED_DIR = Path("data/processed")


class BM25Retriever:
    def __init__(self):
        docs = self._load_documents()
        self.documents = docs
        tokenized = [d["text"].lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def _load_documents(self) -> list[dict]:
        path = PROCESSED_DIR / "documents.json"
        with open(path) as f:
            return json.load(f)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append({
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "title": doc["title"],
                "published": doc["published"],
                "chunk_index": doc["chunk_index"],
                "total_chunks": doc["total_chunks"],
                "score": float(scores[idx]),
                "retriever": "bm25",
            })
        return results
