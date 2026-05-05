import wandb
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "adaptive-rag-xai"


def init_run(run_name: str, config: dict):
    wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        config=config,
    )


def log_retrieval(query: str, query_type: str, strategy: str, chunks: list[dict]):
    wandb.log({
        "retrieval/query_type": query_type,
        "retrieval/strategy": strategy,
        "retrieval/num_chunks": len(chunks),
        "retrieval/top_score": chunks[0]["score"] if chunks else 0,
        "retrieval/avg_score": sum(c["score"] for c in chunks) / len(chunks) if chunks else 0,
    })


def log_xai(explained_chunks: list[dict]):
    top = explained_chunks[0]
    wandb.log({
        "xai/top_chunk_contribution": top["xai_contribution"],
        "xai/top_chunk_title": top["title"][:60],
        "xai/chunks_with_zero_contribution": sum(
            1 for c in explained_chunks if c["xai_contribution"] == 0
        ),
    })


def log_ragas(scores: dict):
    wandb.log({f"ragas/{k}": v for k, v in scores.items() if v is not None})


def finish():
    wandb.finish()
