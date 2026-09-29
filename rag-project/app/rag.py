import json
import os
from typing import Any
import requests

from .database import initialize_database, replace_source_chunks, search_vector

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")


def _ollama_post(path: str, payload: dict[str, Any], stream: bool = False):
    try:
        response = requests.post(f"{OLLAMA_URL}{path}", json=payload, stream=stream, timeout=60)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. Is Ollama running?"
        ) from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
        "keep_alive": "30m",
    }
    resp = _ollama_post("/api/embed", payload).json()
    embeddings = resp.get("embeddings")
    if not embeddings:
        raise RuntimeError("Ollama returned no embeddings.")
    return embeddings


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = end - overlap
    return chunks


def ingest_text(source: str, text: str) -> int:
    chunks = split_text(text)
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)
    initialize_database()
    return replace_source_chunks(
        source,
        [
            {"chunk_index": index, "content": chunk, "embedding": embedding}
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ],
    )


def retrieve(query: str, limit: int = 2) -> list[dict[str, Any]]:
    query_embedding = embed_texts([query])[0]
    return search_vector(query_embedding, limit)


def answer_query(query: str, limit: int = 2) -> dict[str, Any]:
    matches = retrieve(query, limit)
    SIMILARITY_THRESHOLD = 0.40
    valid_matches = [m for m in matches if m.get("score", 0) >= SIMILARITY_THRESHOLD]

    if valid_matches:
        context = "\n\n".join(f"[{m['source']}]: {m['content']}" for m in valid_matches)
        prompt = (
            "You are a helpful, conversational AI assistant. "
            "Use the provided context to answer questions when applicable, otherwise chat naturally.\n\n"
            f"Context:\n{context}\n\n"
            f"User: {query}\n"
            "Assistant:"
        )
        sources_to_return = valid_matches
    else:
        prompt = (
            "You are a friendly, intelligent AI assistant. Respond conversationally and naturally.\n\n"
            f"User: {query}\n"
            "Assistant:"
        )
        sources_to_return = []

    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "num_predict": 180,
            "temperature": 0.2,
            "num_ctx": 1024,
            "num_thread": 6
        },
    }

    res = _ollama_post("/api/chat", payload).json()
    answer = res.get("message", {}).get("content", "").strip()

    return {
        "answer": answer,
        "sources": sources_to_return,
    }


def stream_answer_query(query: str, limit: int = 2):
    matches = retrieve(query, limit)
    SIMILARITY_THRESHOLD = 0.55
    valid_matches = [m for m in matches if m.get("score", 0) >= SIMILARITY_THRESHOLD]

    if valid_matches:
        context = "\n\n".join(f"[{m['source']}]: {m['content']}" for m in valid_matches)
        prompt = (
            "You are a helpful assistant. Answer the user's question accurately using the provided context. "
            "Always conclude your response with a complete final sentence.\n\n"
            f"Context:\n{context}\n\n"
            f"User: {query}\n"
            "Assistant:"
        )
        sources_to_return = valid_matches
    else:
        prompt = (
            "You are a helpful, conversational AI assistant. Respond naturally and conclude with a complete thought.\n\n"
            f"User: {query}\n"
            "Assistant:"
        )
        sources_to_return = []

    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "num_predict": 350,       # Ample room to finish sentences naturally
            "temperature": 0.2,       # Keeps output focused and on-topic
            "num_thread": 6,          # Ensures fast multi-core generation
            "num_ctx": 2048           # Accommodates full context and full answer
        },
    }

    raw_response = _ollama_post("/api/chat", payload, stream=True)

    def event_generator():
        for line in raw_response.iter_lines():
            if line:
                chunk_data = json.loads(line)
                token = chunk_data.get("message", {}).get("content", "")
                if token:
                    yield token

    return event_generator, sources_to_return