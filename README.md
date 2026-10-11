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
