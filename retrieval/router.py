import re
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever

SIMPLE_PATTERNS = [
    r"\bwhat is\b",
    r"\bwho is\b",
    r"\bdefine\b",
    r"\bwhen was\b",
    r"\bwhat does .+ stand for\b",
    r"\bmeaning of\b",
]

COMPLEX_PATTERNS = [
    r"\bhow does\b",
    r"\bwhy does\b",
    r"\bexplain\b",
    r"\bcompare\b",
    r"\bwhat is the role of\b",
    r"\brelationship between\b",
    r"\bimpact of\b",
    r"\bdifference between\b",
]


def classify_query(query: str) -> str:
    q = query.lower()
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, q):
            return "complex"
    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, q):
            return "simple"
    return "simple" if len(q.split()) <= 6 else "complex"


def normalize_scores(results: list[dict]) -> list[dict]:
    if not results:
        return results
    scores = [r["score"] for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        for r in results:
            r["score_normalized"] = 1.0
    else:
        for r in results:
            r["score_normalized"] = (r["score"] - min_s) / (max_s - min_s)
    return results


def deduplicate(results: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for r in results:
        if r["chunk_id"] not in seen:
            seen.add(r["chunk_id"])
            unique.append(r)
    return unique


class QueryRouter:
    def __init__(self):
        print("Loading BM25 retriever...")
        self.bm25 = BM25Retriever()
        print("Loading dense retriever...")
        self.dense = DenseRetriever()
        print("Router ready.\n")

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        query_type = classify_query(query)

        if query_type == "simple":
            results = self.bm25.retrieve(query, top_k=top_k)
            strategy = "BM25 (keyword)"
        else:
            dense_results = normalize_scores(
                self.dense.retrieve(query, top_k=top_k)
            )
            bm25_results = normalize_scores(
                self.bm25.retrieve(query, top_k=top_k // 2)
            )

            # weight: 70% dense, 30% BM25
            for r in dense_results:
                r["score"] = round(r["score_normalized"] * 0.7, 4)
            for r in bm25_results:
                r["score"] = round(r["score_normalized"] * 0.3, 4)

            combined = deduplicate(dense_results + bm25_results)
            results = sorted(combined, key=lambda x: x["score"], reverse=True)[:top_k]
            strategy = "Dense + BM25 hybrid (70/30)"

        return {
            "query": query,
            "query_type": query_type,
            "strategy": strategy,
            "results": results,
        }
