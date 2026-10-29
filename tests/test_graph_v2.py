import os
from unittest.mock import patch

os.environ["AGENT_FAKE_LLM"] = "1"

from agent.graph_v2 import MAX_ITERATIONS, build_graph  # noqa: E402
from agent.schemas import ICP, ResearchDecision  # noqa: E402


class StubSession:
    """Duck-types BrowserSession.fetch without touching real Playwright/network."""

    def __init__(self, link_graph: dict[str, list[str]]):
        self.link_graph = link_graph
        self.fetched: list[str] = []

    def fetch(self, url: str):
        self.fetched.append(url)
        links = self.link_graph.get(url, [])
        return f"text for {url}", links, None


def test_v2_loop_cycles_then_proceeds_to_extract_match_outreach():
    link_graph = {
        "https://example.com": ["https://example.com/about", "https://example.com/careers"],
        "https://example.com/about": [],
        "https://example.com/careers": [],
    }
    session = StubSession(link_graph)

    graph = build_graph(session)
    result = graph.invoke({"url": "https://example.com", "icp": ICP(), "errors": []})

    assert len(session.fetched) > 1, "should have followed at least one candidate link, not just the homepage"
    assert result["company"] is not None
    assert result["match_result"] is not None
    assert result["outreach"] is not None
    assert result["outreach"].status == "draft_needs_human_review"
    assert result["errors"] == []


def test_v2_hard_iteration_cap_stops_an_agent_that_never_wants_to_finish():
    # An ever-expanding link graph: the agent would loop forever if left to its own judgment.
    def infinite_links(url: str) -> list[str]:
        return [f"{url}/next"]

    class InfiniteStubSession:
        def __init__(self):
            self.fetched: list[str] = []

        def fetch(self, url: str):
            self.fetched.append(url)
            return f"text for {url}", infinite_links(url), None

    session = InfiniteStubSession()

    always_scrape = ResearchDecision(action="scrape", url=None, reasoning="never finishes")

    def fake_decide_that_never_finishes(state):
        candidates = state.get("candidate_links", [])
        return ResearchDecision(action="scrape", url=candidates[0], reasoning="never finishes") if candidates else always_scrape

    with patch("agent.graph_v2._fake_decide", side_effect=fake_decide_that_never_finishes):
        graph = build_graph(session)
        result = graph.invoke({"url": "https://example.com", "icp": ICP(), "errors": []})

    assert result["iterations"] == MAX_ITERATIONS
    # Even though research "never finished," the hard cap must still let extract/match/outreach run.
    assert result["company"] is not None
    assert result["outreach"] is not None


def test_v2_degrades_gracefully_when_homepage_unreachable():
    class FailingSession:
        def fetch(self, url: str):
            return "", [], "dns error"

    graph = build_graph(FailingSession())
    result = graph.invoke({"url": "https://nope.invalid", "icp": ICP(), "errors": []})

    assert result.get("company") is None
    assert result.get("outreach") is None
    assert "dns error" in result["errors"]
