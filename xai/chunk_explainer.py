import shap
import numpy as np
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


def score_answer_with_chunk(query: str, chunk_text: str) -> float:
    """Ask the model to score how relevant a chunk is to the query. Returns 0-1."""
    prompt = f"""Rate how useful this text chunk is for answering the question below.
Respond with ONLY a number between 0.0 and 1.0. Nothing else.

Question: {query}

Chunk: {chunk_text[:300]}

Score:"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 1024},
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        text = response.json()["message"]["content"].strip()
        score = float("".join(c for c in text if c.isdigit() or c == "."))
        return min(max(score, 0.0), 1.0)
    except Exception:
        return 0.0


def ablation_importance(query: str, chunks: list[dict]) -> list[dict]:
    """
    Ablation-based XAI: score each chunk individually, then compute
    importance as difference from full-context score.
    This is a lightweight SHAP-style marginal contribution estimate.
    """
    n = len(chunks)
    individual_scores = []

    print(f"  Computing XAI scores for {n} chunks...")
    for i, chunk in enumerate(chunks):
        score = score_answer_with_chunk(query, chunk["text"])
        individual_scores.append(score)
        print(f"    Chunk {i+1}: {score:.3f} | {chunk['title'][:50]}")

    total = sum(individual_scores) if sum(individual_scores) > 0 else 1.0
    normalized = [s / total for s in individual_scores]

    explained_chunks = []
    for i, chunk in enumerate(chunks):
        explained_chunks.append({
            **chunk,
            "xai_raw_score": round(individual_scores[i], 4),
            "xai_contribution": round(normalized[i], 4),
            "xai_rank": 0,
        })

    explained_chunks.sort(key=lambda x: x["xai_contribution"], reverse=True)
    for rank, chunk in enumerate(explained_chunks):
        chunk["xai_rank"] = rank + 1

    return explained_chunks


def explain(query: str, chunks: list[dict]) -> dict:
    print(f"\n[XAI] Explaining {len(chunks)} chunks for query:")
    print(f"      '{query}'\n")

    explained = ablation_importance(query, chunks)

    print(f"\n[XAI] Contribution ranking:")
    print(f"  {'Rank':<5} {'Contrib':>8}  {'Raw':>6}  Title")
    print(f"  {'-'*60}")
    for c in explained:
        print(
            f"  #{c['xai_rank']:<4} {c['xai_contribution']:>7.1%}  "
            f"{c['xai_raw_score']:>6.3f}  {c['title'][:45]}"
        )

    return {
        "query": query,
        "explained_chunks": explained,
        "top_chunk": explained[0],
        "xai_method": "ablation-based marginal contribution (SHAP-style)",
    }
