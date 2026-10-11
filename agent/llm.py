from __future__ import annotations

import os


def fake_mode() -> bool:
    """When set, skip real API calls and use deterministic stubs.

    Lets the graph wiring (scrape -> extract -> match -> END) be exercised and
    tested without an OpenAI API key or incurring API costs. Real runs just
    unset AGENT_FAKE_LLM (or set it to "0") once OPENAI_API_KEY is configured.
    """
    return os.environ.get("AGENT_FAKE_LLM", "0") == "1"
