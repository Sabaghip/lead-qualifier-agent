# Lead Qualifier Agent

A small LangGraph agent that takes a company URL, scrapes the site, extracts a
structured company profile with an LLM, and scores it against an Ideal
Customer Profile (ICP) — the kind of research → matching → pre-qualification
pipeline that's useful for sales/outreach automation.

```
scrape (Playwright) → extract (LLM, structured output) → match (LLM, ICP scoring) → result
```

## Stack

- **LangGraph** for the pipeline (3-node `StateGraph`: `scrape` → `extract` → `match`)
- **LangChain + OpenAI (`gpt-4o-mini`)** with `with_structured_output` for both
  extraction and scoring, so outputs are typed Pydantic models, not free text
- **Playwright** (headless Chromium) for scraping, since most real company
  sites are JS-rendered and plain `requests`/`bs4` returns an empty shell
- **Pydantic** for the shared schemas (`CompanyProfile`, `ICP`, `MatchResult`)

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium

export OPENAI_API_KEY=sk-...
python -m agent.cli https://example.com --icp icp.example.yaml
```

Without an API key, set `AGENT_FAKE_LLM=1` to run the full graph with
deterministic stubs instead of real LLM calls — useful for testing the
scrape → extract → match wiring, error handling, and CLI without burning
API credits (this is how the pipeline was smoke-tested while building it).

## What went wrong, and how I fixed it

1. **Link discovery picked up noise.** A first pass just followed every
   same-domain link found on the homepage, which on a real site (tested
   against anthropic.com) means dozens of blog posts and nav links — way too
   much low-signal text to feed an extraction prompt. Fixed by scoring links
   by keyword hits (`about`, `team`, `product`, `pricing`, `careers`, ...)
   and only following the top matches, capped at `MAX_PAGES = 4` total pages.

2. **Dead or unreachable sites crashed the pipeline.** Pointing the scraper
   at a non-resolving domain raised a Playwright error that killed the whole
   run before `extract`/`match` ever got a chance to run. Fixed by catching
   `PlaywrightTimeoutError`/`PlaywrightError` per-page inside `scrape_site`,
   recording them in `state["errors"]`, and having `extract_node`/`match_node`
   check for missing input and short-circuit with a logged error instead of
   throwing — so a bad URL returns a clean JSON result with `errors` populated
   instead of a stack trace.

## Tests

```bash
python -m pytest tests/ -v
```

Graph-level tests mock `scrape_site` and run in `AGENT_FAKE_LLM` mode, so
they're fast and don't need network access or an API key.

## Limitations / next steps

- No retries/backoff on scraping yet (would add `tenacity` for flaky sites)
- ICP matching is single-shot LLM reasoning; a production version would add
  a cheap rule-based pre-filter (size/industry keywords) before spending an
  LLM call on obvious non-matches
- No persistence — each run is stateless; a real outreach pipeline would
  dedupe against previously-seen companies

## v2 (branch `v2-agentic-outreach`): a genuinely agentic loop + RAG + outreach drafting

v1 above is a **prompt-chaining workflow** — a fixed sequence (`scrape → extract
→ match`), not an autonomous agent, even though it's built with LangGraph. v2
closes that gap and the other two gaps v1 had: no outreach step, no RAG.

```
init (fetch homepage)
  → research_step  ──┐  (LLM decides: scrape one more candidate link, or finish)
       ↑              │
       └──────────────┘  (loops back to itself — a real cycle, capped at MAX_ITERATIONS)
  → extract → match → outreach (RAG-grounded icebreaker, tagged draft_needs_human_review)
```

**What's actually new:**

- **`agent/browser_session.py`** — a single Playwright session kept open and
  driven one `fetch()` call at a time, instead of v1's fixed one-shot crawl.
  URLs are normalized (fragment + trailing slash stripped) before dedup —
  found this the hard way when a `#pricing` anchor on the homepage was
  initially treated as a separate page from the homepage itself.
- **`agent/graph_v2.py`** — `research_step` is a real `add_conditional_edges`
  cycle: after each fetch, an LLM call decides `scrape` (pick one more
  candidate link) or `finish`, routing back to itself or forward to
  `extract`. `MAX_ITERATIONS` is a hard cap enforced in the routing function
  itself, independent of what the LLM decides — so an agent that "never
  wants to stop" still terminates (covered by
  `test_v2_hard_iteration_cap_stops_an_agent_that_never_wants_to_finish`).
  There's also a cheap guard before every decide call: if there are no
  candidate links left, skip the LLM call entirely rather than paying for a
  decision with only one possible answer.
- **`agent/rag.py`** — an actual (if hand-rolled, in-memory) retrieval step:
  a tiny seed corpus of `(observed signal → good icebreaker)` pairs gets
  embedded, and the top-k most similar examples to the current company's
  signals are retrieved and used as style grounding before generating a new
  icebreaker. A real system would swap this for pgvector/Chroma/Pinecone —
  the mechanics (embed → cosine similarity → top-k) are identical either way.
- **`agent/outreach.py`** — generates one personalized icebreaker line
  referencing a specific scraped signal, at `temperature=0.4` (not `0` like
  extraction/matching — a little variation is appropriate for writing, not
  for extracting facts). The output schema's `status` field is hardcoded to
  `"draft_needs_human_review"` — it is never wired to anything that sends a
  message. For an *outreach* agent specifically, that's a deliberate
  guardrail, not an oversight: autonomous outbound messaging is a
  reputational/compliance risk, not just a quality number.

**Run it:**

```bash
AGENT_FAKE_LLM=1 python -m agent.cli_v2 https://example.com --icp icp.example.yaml
# or, with a real key:
OPENAI_API_KEY=sk-... python -m agent.cli_v2 https://example.com --icp icp.example.yaml
```

**Tests:** `tests/test_graph_v2.py` covers the loop actually cycling before
proceeding, the hard iteration cap overriding an agent that never wants to
finish, and graceful degradation on an unreachable site — all via a stub
session, no real browser or API calls needed.
