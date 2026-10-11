from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.extractor import extract_company_profile
from agent.matcher import match_against_icp
from agent.schemas import AgentState
from agent.scraper import scrape_site


def scrape_node(state: AgentState) -> dict:
    pages, errors = scrape_site(state["url"])
    return {"pages": pages, "errors": state.get("errors", []) + errors}


def extract_node(state: AgentState) -> dict:
    if not state.get("pages"):
        return {"errors": state.get("errors", []) + ["no pages to extract from, skipping"]}
    company = extract_company_profile(state["url"], state["pages"])
    return {"company": company}


def match_node(state: AgentState) -> dict:
    if not state.get("company"):
        return {"errors": state.get("errors", []) + ["no company profile to match, skipping"]}
    result = match_against_icp(state["company"], state["icp"])
    return {"match_result": result}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("scrape", scrape_node)
    graph.add_node("extract", extract_node)
    graph.add_node("match", match_node)

    graph.set_entry_point("scrape")
    graph.add_edge("scrape", "extract")
    graph.add_edge("extract", "match")
    graph.add_edge("match", END)

    return graph.compile()
