from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class CompanyProfile(BaseModel):
    name: str = Field(description="Company or product name")
    industry: str = Field(description="Primary industry, e.g. 'fintech', 'logistics SaaS'")
    summary: str = Field(description="2-3 sentence summary of what the company does")
    estimated_size: str = Field(
        description="Rough headcount bucket: 'solo/1-10', '11-50', '51-200', '201-1000', '1000+', or 'unknown'"
    )
    signals: list[str] = Field(
        default_factory=list,
        description="Notable buying/qualification signals found on the site, e.g. 'hiring engineers', "
        "'enterprise clients listed', 'raised funding', 'no pricing page'",
    )


class ICP(BaseModel):
    """Ideal Customer Profile - the qualification criteria to match a scraped company against."""

    target_industries: list[str] = Field(default_factory=list)
    target_size_buckets: list[str] = Field(default_factory=list)
    positive_signals: list[str] = Field(default_factory=list)
    negative_signals: list[str] = Field(default_factory=list)
    notes: str = ""


class MatchResult(BaseModel):
    score: int = Field(ge=0, le=100, description="Qualification score, 0 (no fit) to 100 (perfect fit)")
    qualified: bool
    reasoning: str = Field(description="Short explanation of the score, referencing specific signals")


class AgentState(TypedDict, total=False):
    url: str
    icp: ICP
    pages: dict[str, str]
    company: CompanyProfile
    match_result: MatchResult
    errors: list[str]


# --- v2: agentic research loop + outreach drafting -------------------------


class ResearchDecision(BaseModel):
    """What the research agent decides to do next, given what it's found so far."""

    action: Literal["scrape", "finish"] = Field(
        description="'scrape' to fetch one more candidate link, 'finish' if enough has been gathered"
    )
    url: str | None = Field(
        default=None, description="Which candidate URL to scrape next, required when action='scrape'"
    )
    reasoning: str = Field(description="One sentence on why this action was chosen")


class OutreachDraft(BaseModel):
    icebreaker: str = Field(description="One personalized opening line referencing a specific found fact")
    based_on_signal: str = Field(description="Which specific signal/fact the icebreaker references")
    status: Literal["draft_needs_human_review"] = "draft_needs_human_review"


class ResearchState(TypedDict, total=False):
    url: str
    icp: ICP
    pages: dict[str, str]
    visited: list[str]
    candidate_links: list[str]
    iterations: int
    research_done: bool
    company: CompanyProfile
    match_result: MatchResult
    outreach: OutreachDraft
    errors: list[str]
