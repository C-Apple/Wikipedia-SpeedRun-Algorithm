"""Command-line runner for greedy Wikipedia navigation."""

from __future__ import annotations

import argparse
import logging

from src.wiki_navigator import WikipediaClient, navigate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Navigate Wikipedia hyperlinks toward a target page")
    parser.add_argument("start_page", nargs="?", default="Python (programming language)")
    parser.add_argument("target_page", nargs="?", default="Artificial intelligence")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--link-limit", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = WikipediaClient(timeout=args.timeout)
    result = navigate(
        args.start_page,
        args.target_page,
        client=client,
        max_steps=args.max_steps,
        link_limit=args.link_limit,
    )

    for log in result.logs:
        print(
            f"Visited {log.page!r}: {log.links_found} links in "
            f"{log.elapsed_seconds:.3f}s -> {log.selected_link!r} "
            f"(score={log.selected_score})"
        )
    print("Path:", " -> ".join(result.path))
    print(f"Reached target: {result.reached_target} ({result.reason})")


if __name__ == "__main__":
    main()
