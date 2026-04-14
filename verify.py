import sys

print("=== Checking all dependencies ===\n")

checks = {
    "langchain": "langchain",
    "chromadb": "chromadb",
    "sentence-transformers": "sentence_transformers",
    "rank-bm25": "rank_bm25",
    "arxiv": "arxiv",
    "pymupdf": "fitz",
    "mlx-lm": "mlx_lm",
    "transformers": "transformers",
    "shap": "shap",
    "bertviz": "bertviz",
    "ragas": "ragas",
    "evidently": "evidently",
    "wandb": "wandb",
    "mlflow": "mlflow",
    "fastapi": "fastapi",
    "gradio": "gradio",
}

failed = []
for name, module in checks.items():
    try:
        __import__(module)
        print(f"  ✓  {name}")
    except ImportError:
        print(f"  ✗  {name}  <- MISSING")
        failed.append(name)

print("\n=== Checking Ollama ===\n")
import urllib.request
try:
    urllib.request.urlopen("http://localhost:11434")
    print("  ✓  Ollama server running")
except Exception:
    print("  ✗  Ollama not running — run 'ollama serve' in another tab")

print("\n=== Checking Apple MPS ===\n")
import torch
if torch.backends.mps.is_available():
    print("  ✓  MPS (Metal) available — GPU acceleration active")
else:
    print("  ✗  MPS not available — will run on CPU")

if failed:
    print(f"\n{len(failed)} package(s) failed. Run: pip install {' '.join(failed)}")
else:
    print("\nAll checks passed. Ready to build.")
