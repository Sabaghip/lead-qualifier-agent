from __future__ import annotations

from urllib.parse import urljoin, urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

MAX_PAGES = 4
MAX_CHARS_PER_PAGE = 6000
NAV_TIMEOUT_MS = 15_000
RELEVANT_KEYWORDS = (
    "about",
    "team",
    "product",
    "pricing",
    "careers",
    "jobs",
    "customers",
    "solutions",
    "company",
)


def _same_domain(base: str, candidate: str) -> bool:
    return urlparse(base).netloc == urlparse(candidate).netloc


def _discover_links(page, base_url: str) -> list[str]:
    hrefs = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
    scored: list[tuple[int, str]] = []
    seen = set()
    for href in hrefs:
        if not href or href in seen:
            continue
        seen.add(href)
        if not _same_domain(base_url, href):
            continue
        lowered = href.lower()
        hits = sum(1 for kw in RELEVANT_KEYWORDS if kw in lowered)
        if hits:
            scored.append((hits, href))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [href for _, href in scored]


def scrape_site(url: str) -> tuple[dict[str, str], list[str]]:
    """Fetch the homepage plus up to MAX_PAGES-1 relevant subpages.

    Returns (pages, errors). `pages` maps URL -> visible text (truncated).
    Never raises: a dead/blocking site just yields fewer pages and an error entry,
    so the rest of the graph can still run on whatever text it did get.
    """
    pages: dict[str, str] = {}
    errors: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
            )
        )
        page = context.new_page()
        page.set_default_navigation_timeout(NAV_TIMEOUT_MS)

        to_visit = [url]
        visited = set()
        links_from_home: list[str] = []

        while to_visit and len(pages) < MAX_PAGES:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)
            try:
                page.goto(current, wait_until="domcontentloaded")
                page.wait_for_timeout(500)
                text = page.inner_text("body")
                pages[current] = text[:MAX_CHARS_PER_PAGE]
                if current == url:
                    links_from_home = _discover_links(page, url)
                    for link in links_from_home:
                        if link not in visited and link not in to_visit:
                            to_visit.append(link)
            except PlaywrightTimeoutError:
                errors.append(f"timeout loading {current}")
            except PlaywrightError as exc:
                errors.append(f"error loading {current}: {exc}")

        browser.close()

    if not pages:
        errors.append(f"could not retrieve any content from {url}")

    return pages, errors
