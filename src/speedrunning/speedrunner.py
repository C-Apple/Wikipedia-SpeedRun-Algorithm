from __future__ import annotations
import argparse
import logging
import time as t
from typing import Callable

import numpy as np
from src.speedrunning.vector_visualization import EmbeddingProjector3D
from src.training.word_embedding import SkipGramNegSampling
from src.training.save_configs import load_checkpoint, print_vector
from src.config import LINK_LIMIT, LOAD_MODEL_PATH, START_PAGE, END_PAGE
import torch.nn as nn
from dataclasses import dataclass
import re

from src.speedrunning.wiki_navigator import WikipediaClient

@dataclass
class SpeedrunStepResult:
    """Result of one centralized speedrun navigation step."""

    current_page: str
    next_page: str
    page_found: bool
    stopped: bool
    backtracked: bool
    reason: str | None
    links_found: int
    selected_score: float | None
    ranked_links: list
    elapsed_seconds: float

TITLE_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def project_title(
    title: str,
    word_to_id: dict[str, int],
    model,
    projector: EmbeddingProjector3D,
) -> list[float] | None:
    vector = title_to_vector(
        title,
        word_to_id,
        model,
    )

    if vector is None:
        return None

    return projector.project(
        vector.detach().cpu().numpy()
    )

def title_to_vector(title: str, word_to_id: dict, model):
    tokens = TITLE_TOKEN_RE.findall(title.lower())

    ids = [
        word_to_id[token]
        for token in tokens
        if token in word_to_id
    ]

    if not ids:
        return None

    embeddings = model.in_embedding.weight[ids]
    return embeddings.mean(dim=0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Navigate Wikipedia hyperlinks toward a target page")
    parser.add_argument("start_page", nargs="?", default="Python (programming language)")
    parser.add_argument("target_page", nargs="?", default="Artificial intelligence")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--link-limit", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()

def title_embedding(title: str, word_to_id: dict[str, int], embedding_matrix: np.ndarray) -> np.ndarray | None:
    """Average known token embeddings for a page title.

    Averaging title tokens lets the existing SGNS word vectors represent page
    names containing multiple words, and can be trained further using
    title/link co-occurrence pairs.
    """

    vectors = []
    for token in tokenize_title(title):
        idx = word_to_id.get(token)
        if idx is not None and idx < len(embedding_matrix):
            vectors.append(embedding_matrix[idx])
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def tokenize_title(title: str) -> list[str]:
    """Tokenize multi-word page titles such as 'The White House'."""

    return normalize_title(title).split()

def normalize_title(title: str) -> str:
    """Normalize a Wikipedia title for scoring/training."""

    return " ".join(title.replace("_", " ").casefold().split())

def build_title_scorer(
    target_title: str,
    word_to_id: dict[str, int] | None = None,
    embedding_matrix: np.ndarray | None = None,
) -> Callable[[str], float]:
    """Create a candidate scorer using embeddings when possible."""
    target_vec = title_embedding(target_title, word_to_id, embedding_matrix)

    def score(candidate_title: str) -> float:
        print(f"target_vec: {target_vec}")
        if target_vec is not None and word_to_id is not None and embedding_matrix is not None:
            return compare_word_vectors(target_title, candidate_title, embedding_matrix)
        return 0
    return score

def rank_hyperlinks(
    hyperlinks: list[str],
    end_page: str,
    config: dict,
    visited_pages: set[str],
    current_page: str,
) -> list[tuple[str, float]]:
    """Return valid outgoing links ordered from most to least similar."""

    normalized_current = normalize_title(current_page)

    ranked: list[tuple[str, float]] = []

    for page in hyperlinks:
        normalized_page = normalize_title(page)

        if normalized_page == normalized_current:
            continue

        if normalized_page in visited_pages:
            continue

        similarity = compare_titles(
            page,
            end_page,
            config["word_to_id"],
            config["model"],
        )

        ranked.append((page, similarity))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked

def speedrun_step(
    current_page: str,
    end_page: str,
    config,
    page_history: list[str],
    visited_pages: set[str],
    started_at: float,
    wiki_client: WikipediaClient,
    *,
    link_limit: int = LINK_LIMIT,
    scorer=None,
) -> SpeedrunStepResult:
    
    print(
    f"[STEP ENTERED] current={current_page}, target={end_page}",
    flush=True,
)
    print("[STEP] Fetching hyperlinks", flush=True)

    links = wiki_client.get_hyperlinks(
        current_page,
        limit=link_limit,
    )

    print(
        f"[STEP] Received {len(links)} hyperlinks",
        flush=True,
    )

    print("[STEP] Ranking hyperlinks", flush=True)

    ranked = rank_hyperlinks(
        hyperlinks=links,
        end_page=end_page,
        config=config,
        visited_pages=visited_pages,
        current_page=current_page,
    )

    print(
        f"[STEP] Ranking complete: {len(ranked)} candidates",
        flush=True,
    )

    if not links:
        return SpeedrunStepResult(
            current_page=current_page,
            next_page=current_page,
            page_found=False,
            stopped=True,
            backtracked=False,
            reason="no outgoing article links",
            links_found=0,
            selected_score=None,
            ranked_links=[],
            elapsed_seconds=t.perf_counter() - started_at,
        )
    
    ranked = rank_hyperlinks(
        hyperlinks=links,
        end_page=end_page,
        config=config,
        visited_pages=visited_pages,
        current_page=current_page,
    )


    if normalize_title(end_page) in {
        normalize_title(link) for link in links
        }:
            next_page = next(
                link
                for link in links
                if normalize_title(link) == normalize_title(end_page)
            )

            selected_score = float(1.0)
    elif ranked:
        next_page = None
        selected_score = None

        for candidate_page, candidate_score in ranked:
            if normalize_title(candidate_page) not in visited_pages:
                next_page = candidate_page
                selected_score = candidate_score
                break

        if next_page is None:
            # Every link from the current page has already been visited.
            if len(page_history) <= 1:
                return SpeedrunStepResult(
                    current_page=current_page,
                    next_page=current_page,
                    page_found=False,
                    stopped=True,
                    backtracked=False,
                    reason=(
                        "no unvisited links and no page "
                        "to backtrack to"
                    ),
                    links_found=len(links),
                    selected_score=None,
                    ranked_links=ranked,
                    elapsed_seconds=(
                        t.perf_counter() - started_at
                    ),
                )

            # Remove only the dead-end from the active path.
            # It remains in visited_pages, so it cannot be selected again.
            page_history.pop()
            previous_page = page_history[-1]

            print(
                f"[BACKTRACK] Returning to prior page: "
                f"{previous_page}"
            )

            return SpeedrunStepResult(
                current_page=current_page,
                next_page=previous_page,
                page_found=False,
                stopped=False,
                backtracked=True,
                reason=None,
                links_found=len(links),
                selected_score=None,
                ranked_links=ranked,
                elapsed_seconds=(
                    t.perf_counter() - started_at
                ),
            )

    page_found = (
    normalize_title(next_page)
    == normalize_title(end_page)
    )
    elapsed_seconds = (
        t.perf_counter() - started_at
    )

    print(
        f"Visiting page: {next_page}. "
        f"Elapsed Time: {elapsed_seconds:.3f} seconds.\n"
    )

    return SpeedrunStepResult(
        current_page=current_page,
        next_page=next_page,
        page_found=page_found,
        stopped=False,
        backtracked=False,
        reason=(
            "target reached"
            if page_found
            else None
        ),
        links_found=len(links),
        selected_score=selected_score,
        ranked_links=ranked,
        elapsed_seconds=elapsed_seconds,
    )

def speedrun(start_page, end_page, config):
    wiki = WikipediaClient()
    page_history = []
    total_time = t.perf_counter()
    time_history = [total_time]
    current_page = start_page
    page_found = start_page == end_page
    while page_found is False:
        (current_page, elapsed_time, page_found) = speedrun_step(
            current_page=current_page,
            end_page=end_page,
            config=config,
            page_history=page_history,
            start_time=total_time,
            wiki_client=wiki,
        )

        total_time = elapsed_time + total_time
        time_history.append(total_time)

        if page_found:
            print(f"[SUCCESS] Reached {end_page} in {total_time:.3f} seconds.")

        return page_history, time_history
    
def speedrun(
    start_page: str,
    end_page: str,
    config: dict,
    *,
    max_steps: int = 25,
    link_limit: int = LINK_LIMIT,
    wiki_client: WikipediaClient | None = None,
):
    wiki = wiki_client or WikipediaClient()
    started_at = t.perf_counter()

    current_page = start_page
    page_history = [start_page]
    visited_pages = {normalize_title(start_page)}
    time_history = [0.0]

    reached = normalize_title(start_page) == normalize_title(end_page)
    reason = "target reached" if reached else "max steps reached"

    print(f"Start page: {start_page}. End page: {end_page}.")

    for _ in range(max_steps):
        if reached:
            break

        result = speedrun_step(
            current_page=current_page,
            end_page=end_page,
            config=config,
            page_history=page_history,
            visited_pages=visited_pages,
            started_at=started_at,
            wiki_client=wiki,
            link_limit=link_limit,
        )

        time_history.append(result.elapsed_seconds)

        if result.stopped:
            reason = result.reason or "speedrun stopped"
            break

        current_page = result.next_page

        if not result.backtracked:
            page_history.append(current_page)
            visited_pages.add(normalize_title(current_page))

        if result.page_found:
            reached = True
            reason = "target reached"
            break
    else:
        reached = normalize_title(current_page) == normalize_title(end_page)
        if reached:
            reason = "target reached"

    total_seconds = t.perf_counter() - started_at

    if reached:
        print(
            f"[SUCCESS] Reached {end_page} in "
            f"{total_seconds:.3f} seconds."
        )
    else:
        print(f"[FAILED] {reason}.")

    return page_history, time_history


def compare_word_vectors(word_id1: int, word_id2: int, model):
    embedding1 = model.in_embedding.weight[word_id1]
    embedding2 = model.in_embedding.weight[word_id2]
    cosine_similarity = nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return cosine_similarity.item()

def compare_titles(title1: str, title2: str, word_to_id: dict, model):
    vec1 = title_to_vector(title1, word_to_id, model)
    vec2 = title_to_vector(title2, word_to_id, model)

    if vec1 is None or vec2 is None:
        return -1.0

    cosine_similarity = nn.functional.cosine_similarity(
        vec1.unsqueeze(0),
        vec2.unsqueeze(0)
    )

    return cosine_similarity.item()

def find_closest_hyperlink(current_page, end_page, config, wiki_client : WikipediaClient, previous_pages, hyperlinks):

    #get word ids
    max_similarity = -1
    closest_page = None
    for page in hyperlinks:
        if page == current_page:
            continue
        if page in previous_pages:
            continue
        similarity = compare_titles(page, end_page, config['word_to_id'], config['model'])        
        if similarity > max_similarity:
            max_similarity = similarity
            closest_page = page
    return closest_page
    
def main(start_pg=START_PAGE, end_pg=END_PAGE, model_path=LOAD_MODEL_PATH):
    start_page = START_PAGE if start_pg is None else start_pg
    end_page = START_PAGE if end_pg is None else end_pg

    model, word_to_id, id_to_word, model_config, ckpt = load_checkpoint(
        model_path,
        SkipGramNegSampling
    )

    config = {
        "model": model,
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "model_config": model_config,
    }

    speedrun(start_page, end_page, config)

# def test():
#     model, word_to_id, id_to_word, config, ckpt = load_checkpoint(LOAD_MODEL_PATH, SkipGramNegSampling)
#     print("Vocab size:", len(word_to_id))
#     print("Example keys:", list(word_to_id.keys())[:20])
#     print("Has python?", "python" in word_to_id)
#     print("Has Python?", "Python" in word_to_id)

#     word1 = "good"
#     word2 = "great"
#     if word1 in word_to_id and word2 in word_to_id:
#         similarity1 = compare_word_vectors(word1, word2, word_to_id, model)
#         print(f"Cosine similarity between '{word1}' and '{word2}': {similarity1:.4f}")

#     else:
#         print(f"One of the following words ('{word1}' or '{word2}') not found in vocabulary.")

if __name__ == "__main__":
    main()