from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agent.llm import fake_mode
from agent.schemas import CompanyProfile, ICP, MatchResult

MATCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You score how well a company fits an Ideal Customer Profile (ICP). "
            "Be specific in your reasoning: cite which signals matched or were missing. "
            "A company with no clear fit and no matching signals should score low (<30), "
            "not be scored generously out of politeness.",
        ),
        (
            "human",
            "ICP:\n{icp}\n\nCompany profile:\n{company}\n\n"
            "Score this company's fit against the ICP.",
        ),
    ]
)


def _fake_match(company: CompanyProfile, icp: ICP) -> MatchResult:
    return MatchResult(
        score=0,
        qualified=False,
        reasoning=(
            "AGENT_FAKE_LLM stub: no real scoring performed. "
            f"Would compare '{company.name}' against ICP industries {icp.target_industries}."
        ),
    )


def match_against_icp(company: CompanyProfile, icp: ICP, model: str = "gpt-4o-mini") -> MatchResult:
    if fake_mode():
        return _fake_match(company, icp)
    llm = ChatOpenAI(model=model, temperature=0)
    structured_llm = llm.with_structured_output(MatchResult)
    chain = MATCH_PROMPT | structured_llm
    return chain.invoke(
        {
            "icp": icp.model_dump_json(indent=2),
            "company": company.model_dump_json(indent=2),
        }
    )
