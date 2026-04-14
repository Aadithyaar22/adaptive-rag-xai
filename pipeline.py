from retrieval.router import QueryRouter
from generation.generator import generate
from xai.chunk_explainer import explain
from mlops import tracker

router = QueryRouter()
_run_initialized = False


def run(query: str, top_k: int = 5, track: bool = False) -> dict:
    global _run_initialized

    retrieval_output = router.retrieve(query, top_k=top_k)
    chunks = retrieval_output["results"]

    print(f"\n{'='*60}")
    print(f"Query:    {query}")
    print(f"Type:     {retrieval_output['query_type']}")
    print(f"Strategy: {retrieval_output['strategy']}")
    print(f"Chunks:   {len(chunks)} retrieved")
    print(f"{'='*60}\n")

    print("Answer:\n")
    answer = generate(query, chunks, stream=True)

    xai_output = explain(query, chunks)

    print(f"\n[Top contributing chunk]")
    top = xai_output["top_chunk"]
    print(f"  Title:        {top['title']}")
    print(f"  Contribution: {top['xai_contribution']:.1%}")
    print(f"  Text:         {top['text'][:200]}...")

    if track:
        if not _run_initialized:
            tracker.init_run(
                run_name=f"rag-query-{query[:30].replace(' ', '-')}",
                config={
                    "model": "mistral",
                    "embed_model": "BAAI/bge-small-en-v1.5",
                    "top_k": top_k,
                    "retrieval_strategy": retrieval_output["strategy"],
                    "xai_method": xai_output["xai_method"],
                },
            )
            _run_initialized = True

        tracker.log_retrieval(
            query=query,
            query_type=retrieval_output["query_type"],
            strategy=retrieval_output["strategy"],
            chunks=chunks,
        )
        tracker.log_xai(xai_output["explained_chunks"])

    return {
        "query": query,
        "query_type": retrieval_output["query_type"],
        "strategy": retrieval_output["strategy"],
        "chunks": chunks,
        "answer": answer,
        "xai": xai_output,
    }


if __name__ == "__main__":
    result = run(
        "Explain the role of KL divergence in RLHF training",
        track=True,
    )
    tracker.finish()
