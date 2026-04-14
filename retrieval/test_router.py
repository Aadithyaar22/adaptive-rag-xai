from retrieval.router import QueryRouter

router = QueryRouter()

queries = [
    "what is RAG",
    "what does RLHF stand for",
    "how does RAG reduce hallucination in language models",
    "explain the role of KL divergence in RLHF training",
    "compare BM25 and dense retrieval approaches",
]

for query in queries:
    output = router.retrieve(query, top_k=3)
    print(f"Query:    {output['query']}")
    print(f"Type:     {output['query_type']}")
    print(f"Strategy: {output['strategy']}")
    for i, r in enumerate(output["results"]):
        print(f"  [{i+1}] score={r['score']:.3f} | {r['retriever']:<6} | {r['title'][:60]}")
    print()
