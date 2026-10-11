import os
from unittest.mock import patch

os.environ["AGENT_FAKE_LLM"] = "1"

from agent.graph import build_graph  # noqa: E402
from agent.schemas import ICP  # noqa: E402


def test_graph_runs_end_to_end_with_fake_llm():
    fake_pages = {"https://example.com": "Example Inc.\nWe build widgets."}
    with patch("agent.graph.scrape_site", return_value=(fake_pages, [])):
        graph = build_graph()
        result = graph.invoke(
            {"url": "https://example.com", "icp": ICP(target_industries=["widgets"]), "errors": []}
        )

    assert result["company"].name
    assert result["match_result"] is not None
    assert result["errors"] == []


def test_graph_degrades_gracefully_when_scrape_fails():
    with patch("agent.graph.scrape_site", return_value=({}, ["dns error"])):
        graph = build_graph()
        result = graph.invoke({"url": "https://nope.invalid", "icp": ICP(), "errors": []})

    assert result.get("company") is None
    assert result.get("match_result") is None
    assert "dns error" in result["errors"]
