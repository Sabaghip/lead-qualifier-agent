from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langsmith import traceable

from agent.browser_session import BrowserSession
from agent.extractor import extract_company_profile
from agent.llm import fake_mode
from agent.matcher import match_against_icp
from agent.outreach import generate_outreach_draft
from agent.schemas import ResearchDecision, ResearchState

MAX_ITERATIONS = 4

DECIDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research agent deciding whether to look at one more page of a company's "
            "site, or stop and move to analysis. Only choose to scrape a link if it looks likely "
            "to reveal new, useful information (team size, product, pricing, hiring, customers). "
            "Don't scrape near-duplicate or clearly irrelevant links (e.g. legal/privacy pages). "
            "Finish once you have enough to describe the company and judge its qualification "
            "signals, or if no remaining candidate link looks promising.",
        ),
        (
            "human",
            "Pages already scraped:\n{visited_summary}\n\n"
            "Candidate links not yet visited:\n{candidates}\n\n"
            "Iterations used so far: {iterations} / {max_iterations}\n\n"
            "Decide the next action.",
        ),
    ]
)


def _fake_decide(state: ResearchState) -> ResearchDecision:
    candidates = state.get("candidate_links", [])
    if state.get("iterations", 0) >= 2 or not candidates:
        return ResearchDecision(action="finish", reasoning="fake-mode: stopping after 2 iterations")
    return ResearchDecision(action="scrape", url=candidates[0], reasoning="fake-mode: scraping next candidate")


@traceable(name="research_decide", run_type="chain")
def _real_decide(state: ResearchState) -> ResearchDecision:
    visited_summary = "\n".join(f"- {url}: {text[:150]}..." for url, text in state.get("pages", {}).items())
    candidates = state.get("candidate_links", [])[:10]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ResearchDecision)
    chain = DECIDE_PROMPT | structured_llm
    return chain.invoke(
        {
            "visited_summary": visited_summary or "(none yet)",
            "candidates": "\n".join(candidates) or "(none)",
            "iterations": state.get("iterations", 0),
            "max_iterations": MAX_ITERATIONS,
        }
    )


def build_graph(session: BrowserSession):
    """Builds the v2 agentic graph, bound to one live browser session.

    Unlike v1's fixed scrape -> extract -> match chain, this has a real cycle:
    research_step decides (via the LLM) whether to fetch one more page or stop,
    looping back on itself until it's done or MAX_ITERATIONS is hit, then
    proceeds to extract -> match -> outreach.
    """

    @traceable(name="init", run_type="chain")
    def init_node(state: ResearchState) -> dict:
        text, links, err = session.fetch(state["url"])
        errors = (state.get("errors") or []) + ([err] if err else [])
        return {
            "pages": {state["url"]: text} if text else {},
            "visited": [state["url"]],
            "candidate_links": links,
            "iterations": 0,
            "research_done": False,
            "errors": errors,
        }

    @traceable(name="research_step", run_type="chain")
    def research_step_node(state: ResearchState) -> dict:
        # Cheap check before spending an LLM call: nothing left to look at.
        if not state.get("candidate_links"):
            return {"research_done": True}

        decision = _fake_decide(state) if fake_mode() else _real_decide(state)

        if decision.action == "finish" or not decision.url:
            return {"research_done": True}

        url = decision.url
        text, links, err = session.fetch(url)

        pages = dict(state.get("pages", {}))
        visited = list(state.get("visited", []))
        candidates = [link for link in state.get("candidate_links", []) if link != url]
        errors = list(state.get("errors", []))

        if text:
            pages[url] = text
        visited.append(url)
        for link in links:
            if link not in visited and link not in candidates:
                candidates.append(link)
        if err:
            errors.append(err)

        return {
            "pages": pages,
            "visited": visited,
            "candidate_links": candidates,
            "iterations": state.get("iterations", 0) + 1,
            "research_done": False,
            "errors": errors,
        }

    def route_after_research(state: ResearchState) -> str:
        if state.get("research_done") or state.get("iterations", 0) >= MAX_ITERATIONS:
            return "extract"
        return "research_step"

    @traceable(name="extract", run_type="chain")
    def extract_node(state: ResearchState) -> dict:
        if not state.get("pages"):
            return {"errors": state.get("errors", []) + ["no pages to extract from, skipping"]}
        company = extract_company_profile(state["url"], state["pages"])
        return {"company": company}

    @traceable(name="match", run_type="chain")
    def match_node(state: ResearchState) -> dict:
        if not state.get("company"):
            return {"errors": state.get("errors", []) + ["no company profile to match, skipping"]}
        result = match_against_icp(state["company"], state["icp"])
        return {"match_result": result}

    @traceable(name="outreach", run_type="chain")
    def outreach_node(state: ResearchState) -> dict:
        if not state.get("company") or not state.get("match_result"):
            return {"errors": state.get("errors", []) + ["no company/match to draft outreach from, skipping"]}
        draft = generate_outreach_draft(state["company"], state["match_result"])
        return {"outreach": draft}

    graph = StateGraph(ResearchState)
    graph.add_node("init", init_node)
    graph.add_node("research_step", research_step_node)
    graph.add_node("extract", extract_node)
    graph.add_node("match", match_node)
    graph.add_node("outreach", outreach_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "research_step")
    graph.add_conditional_edges(
        "research_step",
        route_after_research,
        {"extract": "extract", "research_step": "research_step"},
    )
    graph.add_edge("extract", "match")
    graph.add_edge("match", "outreach")
    graph.add_edge("outreach", END)

    return graph.compile()
