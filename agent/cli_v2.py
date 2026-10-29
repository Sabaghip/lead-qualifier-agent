from __future__ import annotations

from dotenv import load_dotenv

# Must run before importing anything LangChain/LangGraph/LangSmith-related.
# langsmith.utils.get_env_var() is @lru_cache'd, and something in that import
# chain calls it (to check whether tracing is enabled) before this module's
# own imports finish - so if load_dotenv() runs after those imports (e.g.
# inside main()), the cache gets primed with pre-.env empty values and
# LANGSMITH_TRACING/LANGSMITH_PROJECT from .env are silently ignored for the
# rest of the process, even though os.environ does get updated correctly.
load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

from agent.browser_session import BrowserSession  # noqa: E402
from agent.cli import load_icp  # noqa: E402
from agent.graph_v2 import build_graph  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic research -> qualify -> draft outreach pipeline (v2)."
    )
    parser.add_argument("url", help="Company homepage URL, e.g. https://example.com")
    parser.add_argument("--icp", default="icp.example.yaml", help="Path to an ICP YAML file")
    args = parser.parse_args()

    icp = load_icp(args.icp)

    with BrowserSession(args.url) as session:
        graph = build_graph(session)
        result = graph.invoke({"url": args.url, "icp": icp, "errors": []})

    output = {
        "url": args.url,
        "iterations": result.get("iterations", 0),
        "visited": result.get("visited", []),
        "company": result["company"].model_dump() if result.get("company") else None,
        "match_result": result["match_result"].model_dump() if result.get("match_result") else None,
        "outreach": result["outreach"].model_dump() if result.get("outreach") else None,
        "errors": result.get("errors", []),
    }
    json.dump(output, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
