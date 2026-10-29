from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agent.llm import fake_mode
from agent.rag import ExampleBank
from agent.schemas import CompanyProfile, MatchResult, OutreachDraft

OUTREACH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write a single personalized cold-outreach icebreaker line, grounded in a specific "
            "signal found about the company - never generic flattery. Use the example icebreakers "
            "only as a style reference, not as content to copy. Output is always a draft that a human "
            "will review before sending, never claim it has been sent.",
        ),
        (
            "human",
            "Company profile:\n{company}\n\nMatch reasoning:\n{match_reasoning}\n\n"
            "Similar past examples (style reference only):\n{examples}\n\n"
            "Write one icebreaker line referencing a specific signal from this company.",
        ),
    ]
)


def _format_examples(examples: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"- Signal: {fact}\n  Icebreaker: {ice}" for fact, ice in examples)


def _fake_outreach(company: CompanyProfile) -> OutreachDraft:
    signal = company.signals[0] if company.signals else "no specific signal found"
    return OutreachDraft(
        icebreaker=f"[fake-mode draft] Noticed: {signal}",
        based_on_signal=signal,
    )


def generate_outreach_draft(company: CompanyProfile, match: MatchResult, model: str = "gpt-4o-mini") -> OutreachDraft:
    if fake_mode():
        return _fake_outreach(company)

    bank = ExampleBank()
    query = " ".join(company.signals) or company.summary
    examples = bank.retrieve(query, k=2)

    llm = ChatOpenAI(model=model, temperature=0.4)
    structured_llm = llm.with_structured_output(OutreachDraft)
    chain = OUTREACH_PROMPT | structured_llm
    return chain.invoke(
        {
            "company": company.model_dump_json(indent=2),
            "match_reasoning": match.reasoning,
            "examples": _format_examples(examples),
        }
    )
