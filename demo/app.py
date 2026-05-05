import gradio as gr
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pipeline import run

def query_pipeline(question: str, top_k: int):
    if not question.strip():
        return "Please enter a question.", "", "", ""

    result = run(question.strip(), top_k=int(top_k), track=False)

    answer = result["answer"]

    sources_lines = []
    for i, chunk in enumerate(result["chunks"]):
        sources_lines.append(
            f"[{i+1}] {chunk['title']} ({chunk['published']}) — score: {chunk['score']:.3f}"
        )
    sources = "\n".join(sources_lines)

    xai_lines = []
    for chunk in result["xai"]["explained_chunks"]:
        bar = "█" * int(chunk["xai_contribution"] * 20)
        xai_lines.append(
            f"#{chunk['xai_rank']} {bar} {chunk['xai_contribution']:.1%}\n"
            f"    {chunk['title'][:70]}"
        )
    xai = "\n\n".join(xai_lines)

    meta = (
        f"Query type:  {result['query_type']}\n"
        f"Strategy:    {result['strategy']}\n"
        f"XAI method:  {result['xai']['xai_method']}\n"
        f"Chunks used: {len(result['chunks'])}"
    )

    return answer, sources, xai, meta


EXAMPLE_QUERIES = [
    "How does RAG reduce hallucination in language models?",
    "What is the role of KL divergence in RLHF training?",
    "What is the difference between RLAIF and RLHF?",
    "Explain how attention mechanisms work in transformers",
    "What are the limitations of fine-tuning large language models?",
]

with gr.Blocks(title="Adaptive RAG + XAI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# Adaptive RAG with Explainable AI
**ArXiv ML Papers QA System** — Dense + BM25 hybrid retrieval, Mistral-7B generation, SHAP-style chunk attribution.

Built on 50 ArXiv papers across RAG, RLHF, transformers, fine-tuning, and XAI.
""")

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Ask a question about ML / AI",
                placeholder="e.g. How does RAG reduce hallucination?",
                lines=2,
            )
            top_k = gr.Slider(
                minimum=3,
                maximum=10,
                value=5,
                step=1,
                label="Number of chunks to retrieve",
            )
            submit_btn = gr.Button("Ask", variant="primary")

        with gr.Column(scale=1):
            meta_box = gr.Textbox(
                label="Pipeline metadata",
                lines=5,
                interactive=False,
            )

    answer_box = gr.Textbox(
        label="Answer",
        lines=8,
        interactive=False,
    )

    with gr.Row():
        with gr.Column():
            sources_box = gr.Textbox(
                label="Retrieved sources",
                lines=8,
                interactive=False,
            )
        with gr.Column():
            xai_box = gr.Textbox(
                label="XAI — chunk contribution ranking",
                lines=8,
                interactive=False,
            )

    gr.Examples(
        examples=EXAMPLE_QUERIES,
        inputs=question,
        label="Example queries",
    )

    submit_btn.click(
        fn=query_pipeline,
        inputs=[question, top_k],
        outputs=[answer_box, sources_box, xai_box, meta_box],
    )
    question.submit(
        fn=query_pipeline,
        inputs=[question, top_k],
        outputs=[answer_box, sources_box, xai_box, meta_box],
    )

if __name__ == "__main__":
    demo.launch(share=True)
