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

import time
from src.word_embedding import SkipGramNegSampling
from src.save_configs import load_checkpoint, print_vector
from src.constants import LOAD_MODEL_PATH, START_PAGE, END_PAGE
import torch.nn as nn


#load nn config from file, pretrained

def run(start_page, end_page, config):
    page_history = []
    t0 = time.perf_counter()
    t1 = time.perf_counter()
    current_page = start_page
    page_history.append(current_page)
    pages_visited = 0
    while current_page != end_page:
        closest_hyperlink = find_closest_hyperlink(current_page, end_page, config)
        t1 = time.perf_counter()
        print(f"Visiting page: {closest_hyperlink}. Time: {t1 - t0} \n")
        page_history.append(closest_hyperlink)
        pages_visited += 1
    t1 = time.perf_counter()
    timer = t1 - t0
    return page_history, timer

def compare_word_vectors(word_id1: int, word_id2: int, model):
    embedding1 = model.in_embedding.weight[word_id1]
    embedding2 = model.in_embedding.weight[word_id2]
    cosine_similarity = nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return cosine_similarity.item()

def compare_word_vectors(word_id1: str, word_id2: str, word_to_id: dict, model):
    word_id1 = word_id1.lower()
    word_id2 = word_id2.lower()
    word_id1 = word_to_id[word_id1]
    word_id2 = word_to_id[word_id2]

    embedding1 = model.in_embedding.weight[word_id1]
    embedding2 = model.in_embedding.weight[word_id2]
    cosine_similarity = nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return cosine_similarity.item()

def find_closest_hyperlink(current_page, end_page, config):
    hyperlinks = [] #TODO find hyperlinks of current page
    #get word ids
    max_similarity = -1
    closest_page = None
    for page in hyperlinks:
        if page == current_page:
            continue
        similarity = compare_word_vectors(page, end_page, config['word_to_id'], config['model'])
        if similarity > max_similarity:
            max_similarity = similarity
            closest_page = page
    return closest_page
    
def main():
    start_page = START_PAGE
    end_page = END_PAGE
    config = {} #load config from file
    run(start_page, end_page, config)

def test():
    model, word_to_id, id_to_word, config, ckpt = load_checkpoint(LOAD_MODEL_PATH, SkipGramNegSampling)
    print("Vocab size:", len(word_to_id))
    print("Example keys:", list(word_to_id.keys())[:20])
    print("Has python?", "python" in word_to_id)
    print("Has Python?", "Python" in word_to_id)

    word1 = "good"
    word2 = "great"
    if word1 in word_to_id and word2 in word_to_id:
        similarity1 = compare_word_vectors(word1, word2, word_to_id, model)
        print(f"Cosine similarity between '{word1}' and '{word2}': {similarity1:.4f}")

    else:
        print(f"One of the following words ('{word1}' or '{word2}') not found in vocabulary.")

if __name__ == "__main__":
    test()