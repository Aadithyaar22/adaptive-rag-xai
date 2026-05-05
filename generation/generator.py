from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

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

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1000,
        stream=stream,
    )

    full_response = ""
    if stream:
        for chunk in response:
            token = chunk.choices[0].delta.content or ""
            print(token, end="", flush=True)
            full_response += token
        print()
    else:
        full_response = response.choices[0].message.content

    return full_response
