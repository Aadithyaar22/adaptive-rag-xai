import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHROMA_DIR = Path("data/chroma")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "arxiv_ml_papers"

def retrieve(query: str, top_k: int = 5):
    embedder = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embedder.encode(
        query, normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nQuery: '{query}'\n")
    print(f"Top {top_k} results:\n")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        score = round(1 - dist, 3)
        print(f"[{i+1}] Score: {score}")
        print(f"     Title: {meta['title'][:80]}")
        print(f"     Published: {meta['published']}")
        print(f"     Chunk: {meta['chunk_index']}/{meta['total_chunks']}")
        print(f"     Text: {doc[:150]}...")
        print()

if __name__ == "__main__":
    retrieve("how does RAG reduce hallucination in language models")
    retrieve("what is the role of KL divergence in RLHF")
