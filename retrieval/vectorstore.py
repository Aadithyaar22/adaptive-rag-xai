import json
import chromadb
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path("data/processed")
CHROMA_DIR = Path("data/chroma")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "arxiv_ml_papers"
BATCH_SIZE = 64


def load_documents() -> list[dict]:
    path = PROCESSED_DIR / "documents.json"
    with open(path) as f:
        return json.load(f)


def build_vectorstore():
    print("Loading documents...")
    docs = load_documents()
    print(f"  {len(docs)} chunks loaded")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("\nConnecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"  Collection '{COLLECTION_NAME}' exists — deleting and rebuilding")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\nEmbedding and inserting {len(docs)} chunks in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(docs), BATCH_SIZE)):
        batch = docs[i : i + BATCH_SIZE]

        texts = [d["text"] for d in batch]
        ids = [d["chunk_id"] for d in batch]
        metadatas = [
            {
                "paper_id": d["paper_id"],
                "title": d["title"],
                "published": d["published"],
                "chunk_index": d["chunk_index"],
                "total_chunks": d["total_chunks"],
                "abstract": d["abstract"][:500],
            }
            for d in batch
        ]

        embeddings = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"\n{'='*40}")
    print(f"Chunks embedded:  {collection.count()}")
    print(f"Embedding model:  {EMBED_MODEL}")
    print(f"Stored at:        {CHROMA_DIR}")
    print(f"{'='*40}")


if __name__ == "__main__":
    build_vectorstore()
