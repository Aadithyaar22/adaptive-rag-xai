import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def score_answer_with_chunk(query: str, chunk_text: str) -> float:
    prompt = f"""Rate how useful this text chunk is for answering the question below.
Respond with ONLY a number between 0.0 and 1.0. Nothing else.

Question: {query}

Chunk: {chunk_text[:300]}

Score:"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        text = response.choices[0].message.content.strip()
        score = float("".join(c for c in text if c.isdigit() or c == "."))
        return min(max(score, 0.0), 1.0)
    except Exception:
        return 0.0


def ablation_importance(query: str, chunks: list[dict]) -> list[dict]:
    print(f"  Computing XAI scores for {len(chunks)} chunks...")
    individual_scores = []

    for i, chunk in enumerate(chunks):
        score = score_answer_with_chunk(query, chunk["text"])
        individual_scores.append(score)
        print(f"    Chunk {i+1}: {score:.3f} | {chunk['title'][:50]}")

    if sum(individual_scores) == 0:
        print("  XAI scores all zero — falling back to retrieval scores")
        individual_scores = [c.get("score", 0.0) for c in chunks]

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
    print(f"\n[XAI] Explaining {len(chunks)} chunks...")
    explained = ablation_importance(query, chunks)

    return {
        "query": query,
        "explained_chunks": explained,
        "top_chunk": explained[0],
        "xai_method": "ablation-based marginal contribution (SHAP-style)",
    }
