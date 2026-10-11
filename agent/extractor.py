from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agent.llm import fake_mode
from agent.schemas import CompanyProfile

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured company information from raw scraped website text. "
            "Base every field only on what's actually in the text. If something isn't stated, "
            "say so plainly (e.g. estimated_size='unknown') rather than guessing.",
        ),
        (
            "human",
            "Scraped pages from {url}:\n\n{combined_text}\n\n"
            "Extract a CompanyProfile from this.",
        ),
    ]
)


def _fake_extract(url: str, pages: dict[str, str]) -> CompanyProfile:
    first_page_text = next(iter(pages.values()), "")
    first_line = next((line.strip() for line in first_page_text.splitlines() if line.strip()), url)
    return CompanyProfile(
        name=first_line[:60],
        industry="unknown (AGENT_FAKE_LLM stub, no real extraction performed)",
        summary=f"Stub profile generated without calling an LLM. Scraped {len(pages)} page(s) from {url}.",
        estimated_size="unknown",
        signals=[f"fake-mode: {len(first_page_text)} chars scraped from homepage"],
    )


def extract_company_profile(url: str, pages: dict[str, str], model: str = "gpt-4o-mini") -> CompanyProfile:
    if fake_mode():
        return _fake_extract(url, pages)
    combined_text = "\n\n---\n\n".join(f"[{page_url}]\n{text}" for page_url, text in pages.items())
    llm = ChatOpenAI(model=model, temperature=0)
    structured_llm = llm.with_structured_output(CompanyProfile)
    chain = EXTRACTION_PROMPT | structured_llm
    return chain.invoke({"url": url, "combined_text": combined_text})
