from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from agent.browser_session import BrowserSession
from agent.cli import load_icp
from agent.graph_v2 import build_graph


def main() -> None:
    load_dotenv()

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
