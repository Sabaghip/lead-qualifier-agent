from __future__ import annotations

import argparse
import json
import sys

import yaml
from dotenv import load_dotenv

from agent.graph import build_graph
from agent.schemas import ICP


def load_icp(path: str) -> ICP:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ICP(**raw)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Scrape a company site and score it against an ICP.")
    parser.add_argument("url", help="Company homepage URL, e.g. https://example.com")
    parser.add_argument("--icp", default="icp.example.yaml", help="Path to an ICP YAML file")
    args = parser.parse_args()

    icp = load_icp(args.icp)
    graph = build_graph()

    result = graph.invoke({"url": args.url, "icp": icp, "errors": []})

    output = {
        "url": args.url,
        "company": result["company"].model_dump() if result.get("company") else None,
        "match_result": result["match_result"].model_dump() if result.get("match_result") else None,
        "pages_scraped": list(result.get("pages", {}).keys()),
        "errors": result.get("errors", []),
    }
    json.dump(output, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
