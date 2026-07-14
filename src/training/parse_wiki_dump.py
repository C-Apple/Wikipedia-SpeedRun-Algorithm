from __future__ import annotations

import argparse
import bz2
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import mwparserfromhell


# Wikipedia XML namespace used in dump files.
MEDIAWIKI_NAMESPACE = "http://www.mediawiki.org/xml/export-0.11/"

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
ENTITY_PREFIX = "wiki::"
WHITESPACE_RE = re.compile(r"\s+")

# These links are not normal encyclopedia article links.
IGNORED_LINK_NAMESPACES = {
    "category",
    "file",
    "image",
    "template",
    "help",
    "portal",
    "wikipedia",
    "special",
    "talk",
    "user",
    "module",
    "media",
    "draft",
    "book",
    "timedtext",
}

def normalize_title(title: str) -> str:
    """
    Normalize a Wikipedia page title while keeping it human-readable.

    Examples:
        "White_House" -> "White House"
        "Chess#History" -> "Chess"
    """
    title = html.unescape(title)
    title = title.split("#", 1)[0]
    title = title.replace("_", " ")
    title = WHITESPACE_RE.sub(" ", title).strip()

    return title

def normalize_entity_token(title: str) -> str:
    """
    Convert a Wikipedia destination title into one trainable token.

    Examples:
        "White House" -> "wiki::white_house"
        "Washington, D.C." -> "wiki::washington_d_c"
    """
    title = normalize_title(title)
    title = title.casefold()

    # Replace punctuation and whitespace with underscores.
    title = re.sub(r"[^a-z0-9]+", "_", title)
    title = title.strip("_")

    return f"{ENTITY_PREFIX}{title}"

def tokenize_plain_text(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def extract_training_tokens(wikitext: str) -> list[str]:
    """
    Convert ordinary article text into word tokens while replacing
    internal article links with atomic destination tokens.

    Example:
        "The [[White House|presidential residence]] is in Washington."

    becomes:
        [
            "the",
            "wiki::white_house",
            "is",
            "in",
            "washington",
        ]
    """
    code = mwparserfromhell.parse(wikitext)
    tokens: list[str] = []

    for node in code.nodes:
        if isinstance(node, mwparserfromhell.nodes.Wikilink):
            target = normalize_title(str(node.title))

            if target and is_article_link(target):
                entity_token = normalize_entity_token(target)

                # Avoid creating an empty token such as "wiki::".
                if entity_token != ENTITY_PREFIX:
                    tokens.append(entity_token)
            else:
                visible_text = (
                    str(node.text)
                    if node.text is not None
                    else str(node.title)
                )

                tokens.extend(tokenize_plain_text(visible_text))

        else:
            try:
                visible_text = node.__strip__()
            except (AttributeError, TypeError):
                visible_text = str(node)

            if visible_text:
                tokens.extend(
                    tokenize_plain_text(str(visible_text))
                )

    return tokens

def is_article_link(title: str) -> bool:
    """
    Return True when a link appears to point to a normal article.

    This excludes links such as:
        File:Example.jpg
        Category:Chess
        Template:Infobox
    """
    title = title.strip()

    # A leading colon suppresses category/file behavior in wikitext.
    title = title.lstrip(":")

    if ":" not in title:
        return True

    namespace = title.split(":", 1)[0].casefold()
    return namespace not in IGNORED_LINK_NAMESPACES


def extract_links(wikitext: str) -> list[str]:
    """
    Extract unique internal Wikipedia article destinations.

    Example:
        [[White House|the White House]]
    becomes:
        "White House"
    """
    code = mwparserfromhell.parse(wikitext)

    links: list[str] = []
    seen: set[str] = set()

    for wikilink in code.filter_wikilinks(recursive=True):
        target = normalize_title(str(wikilink.title))

        if not target:
            continue

        if not is_article_link(target):
            continue

        # Ignore local section-only links such as [[#History]].
        if target.startswith("#"):
            continue

        key = target.casefold()

        if key not in seen:
            seen.add(key)
            links.append(target)

    return links


def extract_clean_text(wikitext: str) -> str:
    """
    Remove most Wikipedia markup and return readable article text.
    """
    code = mwparserfromhell.parse(wikitext)

    # normalize=True resolves some entities and formatting.
    # collapse=True collapses excessive whitespace.
    text = code.strip_code(normalize=True, collapse=True)

    text = html.unescape(text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text


def get_child_text(
    element: ET.Element,
    child_name: str,
    default: str = "",
) -> str:
    child = element.find(f"{{{MEDIAWIKI_NAMESPACE}}}{child_name}")

    if child is None or child.text is None:
        return default

    return child.text


def iter_dump_pages(
    dump_path: Path,
    max_pages: int | None = None,
) -> Iterator[dict]:
    """
    Stream namespace-0 articles from a compressed Wikipedia XML dump.

    The dump is never fully loaded into memory.
    """
    processed_articles = 0

    with bz2.open(dump_path, "rb") as dump_file:
        context = ET.iterparse(dump_file, events=("end",))

        for _, element in context:
            if element.tag != f"{{{MEDIAWIKI_NAMESPACE}}}page":
                continue

            title = get_child_text(element, "title")
            namespace = get_child_text(element, "ns")

            # Namespace 0 contains normal encyclopedia articles.
            if namespace != "0":
                element.clear()
                continue

            page_id = get_child_text(element, "id")

            redirect_element = element.find(
                f"{{{MEDIAWIKI_NAMESPACE}}}redirect"
            )

            revision = element.find(
                f"{{{MEDIAWIKI_NAMESPACE}}}revision"
            )

            if revision is None:
                element.clear()
                continue

            text_element = revision.find(
                f"{{{MEDIAWIKI_NAMESPACE}}}text"
            )

            if text_element is None or not text_element.text:
                element.clear()
                continue

            wikitext = text_element.text

            record = {
                "page_id": int(page_id) if page_id.isdigit() else None,
                "title": title,
                "redirect": (
                    redirect_element.get("title")
                    if redirect_element is not None
                    else None
                ),
                "text": extract_clean_text(wikitext),
                "tokens": extract_training_tokens(wikitext),
                "links": extract_links(wikitext),
            }

            yield record

            processed_articles += 1

            if processed_articles % 1_000 == 0:
                print(
                    f"[INFO] Parsed "
                    f"{processed_articles:,} articles"
                )

            element.clear()

            if (
                max_pages is not None
                and processed_articles >= max_pages
            ):
                break


def parse_dump(
    dump_path: Path,
    output_path: Path,
    max_pages: int | None = None,
) -> None:
    """
    Parse a Wikipedia dump and save the result as JSON Lines.

    Each output line is one independent JSON object.
    """
    dump_path = dump_path.resolve()
    output_path = output_path.resolve()

    if not dump_path.exists():
        raise FileNotFoundError(
            f"Wikipedia dump does not exist: {dump_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    article_count = 0
    link_count = 0

    print(f"[INFO] Reading: {dump_path}")
    print(f"[INFO] Writing: {output_path}")

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in iter_dump_pages(
            dump_path,
            max_pages=max_pages,
        ):
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            article_count += 1
            link_count += len(record["links"])

    print(f"[DONE] Wrote {article_count:,} articles")
    print(f"[DONE] Extracted {link_count:,} article links")

    if article_count:
        print(
            f"[DONE] Average links per article: "
            f"{link_count / article_count:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Wikipedia article text and internal links "
            "from an XML .bz2 dump."
        )
    )

    parser.add_argument(
        "dump_path",
        type=Path,
        help="Path to the Wikipedia .xml.bz2 dump",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/wikipedia_articles.jsonl"),
        help="Output JSONL path",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional article limit for testing",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parse_dump(
        dump_path=args.dump_path,
        output_path=args.output,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()