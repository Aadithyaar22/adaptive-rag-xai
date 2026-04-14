import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

SYSTEM_PROMPT = """You are a research assistant specializing in machine learning and AI.
Answer questions using ONLY the provided context chunks from ArXiv papers.
If the context does not contain enough information, say so clearly.
Always cite which paper your answer draws from.
Be precise and technical — your audience are ML engineers."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(
            f"[{i+1}] From: '{chunk['title']}' ({chunk['published']})\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)
    return f"""Context chunks from ArXiv papers:

{context}

Question: {query}

Answer based strictly on the context above. Cite sources by their title."""


def generate(query: str, chunks: list[dict], stream: bool = True) -> str:
    prompt = build_prompt(query, chunks)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": stream,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 4096,
        },
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        stream=stream,
    )
    response.raise_for_status()

    full_response = ""
    if stream:
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                print(token, end="", flush=True)
                full_response += token
                if data.get("done"):
                    break
        print()
    else:
        data = response.json()
        full_response = data["message"]["content"]

    return full_response
