from pipeline import run
from evaluation.ragas_eval import run_evaluation
from mlops import tracker
import wandb

if __name__ == "__main__":
    tracker.init_run(
        run_name="ragas-evaluation-v1",
        config={
            "model": "mistral",
            "embed_model": "BAAI/bge-small-en-v1.5",
            "retrieval_strategy": "dense+bm25-hybrid-70-30",
            "xai_method": "ablation-marginal-contribution",
            "num_test_samples": 3,
            "top_k": 5,
        },
    )

    results = run_evaluation(pipeline_fn=run)
    df = results.to_pandas()

    scores = {
        "answer_relevancy": df["answer_relevancy"].mean(),
        "context_precision": df["context_precision"].mean(),
    }

    tracker.log_ragas(scores)

    wandb.log({
        "ragas/answer_relevancy": scores["answer_relevancy"],
        "ragas/context_precision": scores["context_precision"],
    })

    tracker.finish()
    print("\nAll metrics logged to W&B.")
