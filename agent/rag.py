from __future__ import annotations

import hashlib
import math

from agent.llm import fake_mode

# A tiny seed corpus of (observed signal -> good icebreaker) pairs. In a real
# system this would grow from every past qualified lead; here it's just
# enough to demonstrate retrieval feeding generation (the actual point of RAG).
EXAMPLE_BANK: list[tuple[str, str]] = [
    (
        "Company is hiring multiple backend engineers and recently announced a funding round.",
        "Noticed you're scaling the backend team fast after the raise - curious how you're handling "
        "the growth in infra complexity.",
    ),
    (
        "Company lists well-known enterprise clients on its homepage.",
        "Saw the enterprise logos on your site - impressive traction. Wondered if lead research and "
        "qualification is still mostly manual for your sales team.",
    ),
    (
        "Company has no public pricing page, only a 'book a demo' call to action.",
        "Noticed pricing's behind a demo call - usually means a more complex sales motion. How's lead "
        "qualification handled today before a call gets booked?",
    ),
    (
        "Company's careers page lists several open AI/ML engineering roles.",
        "Saw you're hiring multiple AI/ML engineers right now - sounds like real investment in agent/LLM "
        "work. Would love to hear what you're building.",
    ),
    (
        "Company's blog or changelog has had no new posts in a long time.",
        "Noticed things have been quiet on the blog lately - often means the team's heads-down building "
        "rather than publishing. Curious what's shipped recently.",
    ),
]


def _fake_embedding(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-words hash embedding, used only when AGENT_FAKE_LLM=1.

    Not semantically meaningful - just lets retrieval be exercised and tested
    without calling a real embeddings API.
    """
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    return vec


def _real_embeddings(texts: list[str]) -> list[list[float]]:
    from langchain_openai import OpenAIEmbeddings

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    return embedder.embed_documents(texts)


def _embed(texts: list[str]) -> list[list[float]]:
    if fake_mode():
        return [_fake_embedding(t) for t in texts]
    return _real_embeddings(texts)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (norm_a * norm_b)


class ExampleBank:
    """Hand-rolled in-memory vector store. A real system would use pgvector/
    Chroma/Pinecone instead - this exists to make the retrieval mechanics
    explicit and dependency-free for a small demo corpus."""

    def __init__(self):
        self._facts = [fact for fact, _ in EXAMPLE_BANK]
        self._icebreakers = [ice for _, ice in EXAMPLE_BANK]
        self._embeddings = _embed(self._facts)

    def retrieve(self, query: str, k: int = 2) -> list[tuple[str, str]]:
        query_vec = _embed([query])[0]
        scored = sorted(
            zip(self._facts, self._icebreakers, self._embeddings),
            key=lambda row: _cosine(query_vec, row[2]),
            reverse=True,
        )
        return [(fact, ice) for fact, ice, _ in scored[:k]]
