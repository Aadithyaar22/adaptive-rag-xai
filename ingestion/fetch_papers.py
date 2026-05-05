import arxiv
import fitz
import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "retrieval augmented generation",
    "large language model fine-tuning",
    "explainable AI machine learning",
    "transformer attention mechanism",
    "RLHF reinforcement learning human feedback",
]

MAX_PAPERS_PER_QUERY = 10
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def fetch_papers(query: str, max_results: int) -> list[dict]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = []
    for result in client.results(search):
        papers.append({
            "id": result.entry_id.split("/")[-1],
            "title": result.title,
            "authors": [a.name for a in result.authors[:5]],
            "abstract": result.summary,
            "published": str(result.published.date()),
            "url": result.pdf_url,
            "result_obj": result,
        })
    return papers


def download_pdf(paper: dict) -> Path | None:
    pdf_path = RAW_DIR / f"{paper['id']}.pdf"
    if pdf_path.exists():
        return pdf_path
    try:
        paper["result_obj"].download_pdf(dirpath=str(RAW_DIR), filename=f"{paper['id']}.pdf")
        time.sleep(1)
        return pdf_path
    except Exception as e:
        print(f"  Failed to download {paper['id']}: {e}")
        return None


def extract_text(pdf_path: Path) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"  Failed to parse {pdf_path.name}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def process_paper(paper: dict, pdf_path: Path) -> list[dict]:
    text = extract_text(pdf_path)
    if not text:
        return []
    chunks = chunk_text(text)
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "chunk_id": f"{paper['id']}_chunk_{i}",
            "paper_id": paper["id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "published": paper["published"],
            "abstract": paper["abstract"],
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text": chunk,
        })
    return documents


def main():
    all_documents = []
    seen_ids = set()

    for query in QUERIES:
        print(f"\nFetching: '{query}'")
        papers = fetch_papers(query, MAX_PAPERS_PER_QUERY)
        print(f"  Found {len(papers)} papers")

        for paper in tqdm(papers, desc="  Processing"):
            if paper["id"] in seen_ids:
                continue
            seen_ids.add(paper["id"])

            pdf_path = download_pdf(paper)
            if not pdf_path:
                continue

            docs = process_paper(paper, pdf_path)
            all_documents.extend(docs)

    output_path = PROCESSED_DIR / "documents.json"
    with open(output_path, "w") as f:
        json.dump(all_documents, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Papers fetched:   {len(seen_ids)}")
    print(f"Total chunks:     {len(all_documents)}")
    print(f"Saved to:         {output_path}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
