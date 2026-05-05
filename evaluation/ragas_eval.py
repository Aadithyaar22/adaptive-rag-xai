import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
RESULTS_DIR = Path("evaluation")

TEST_SAMPLES = [
    {
        "question": "What is the role of KL divergence in RLHF?",
        "ground_truth": "KL divergence in RLHF acts as a regularization constraint that prevents the learned policy from deviating too far from the original reference policy, maintaining response quality and safety during training.",
    },
    {
        "question": "How does RAG reduce hallucination in language models?",
        "ground_truth": "RAG reduces hallucination by grounding model responses in retrieved factual context from external documents, limiting the model to generating answers based on retrieved evidence rather than parametric memory alone.",
    },
    {
        "question": "What is the difference between RLAIF and RLHF?",
        "ground_truth": "RLHF uses human feedback to train a reward model, while RLAIF replaces human annotators with AI feedback, making it more scalable and cheaper while achieving comparable alignment performance.",
    },
]


def build_ragas_dataset(pipeline_fn) -> Dataset:
    questions, answers, contexts, ground_truths = [], [], [], []
    for sample in TEST_SAMPLES:
        print(f"\nRunning pipeline for: {sample['question']}")
        result = pipeline_fn(sample["question"], top_k=5)
        questions.append(sample["question"])
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result["chunks"]])
        ground_truths.append(sample["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def run_evaluation(pipeline_fn):
    print("Building evaluation dataset...")
    dataset = build_ragas_dataset(pipeline_fn)

    print("\nSetting up Ollama LLM + embeddings for RAGAS...")
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        timeout=300,
        format="json",        # force JSON output — fixes parse failures
    )
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    run_config = RunConfig(
        max_workers=1,
        timeout=300,
        max_retries=3,
    )

    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=run_config,
    )

    df = results.to_pandas()

    def safe_mean(col):
        return f"{df[col].mean():.3f}" if not df[col].isna().all() else "n/a"

    print(f"\n{'='*50}")
    print("RAGAS EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Faithfulness:      {safe_mean('faithfulness')}")
    print(f"  Answer Relevancy:  {safe_mean('answer_relevancy')}")
    print(f"  Context Precision: {safe_mean('context_precision')}")
    print(f"  Context Recall:    {safe_mean('context_recall')}")
    print(f"{'='*50}")

    output_path = RESULTS_DIR / "ragas_results.json"
    records = df.to_dict(orient="records")
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"\nPer-sample results saved to {output_path}")

    return results
