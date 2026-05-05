import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path("data/chroma")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "arxiv_ml_papers"


class DenseRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "chunk_id": meta.get("paper_id", "") + f"_chunk_{meta['chunk_index']}",
                "text": doc,
                "title": meta["title"],
                "published": meta["published"],
                "chunk_index": meta["chunk_index"],
                "total_chunks": meta["total_chunks"],
                "score": round(1 - dist, 4),
                "retriever": "dense",
            })
        return output
