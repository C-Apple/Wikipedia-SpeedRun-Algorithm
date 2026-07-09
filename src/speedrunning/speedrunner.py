from __future__ import annotations
import argparse
import logging
import time
from src.training.word_embedding import SkipGramNegSampling
from src.training.save_configs import load_checkpoint, print_vector
from src.config import LOAD_MODEL_PATH, START_PAGE, END_PAGE
import torch.nn as nn
import re

from src.speedrunning.wiki_navigator import WikipediaClient


TITLE_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

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

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Navigate Wikipedia hyperlinks toward a target page")
#     parser.add_argument("start_page", nargs="?", default="Python (programming language)")
#     parser.add_argument("target_page", nargs="?", default="Artificial intelligence")
#     parser.add_argument("--max-steps", type=int, default=25)
#     parser.add_argument("--link-limit", type=int, default=500)
#     parser.add_argument("--timeout", type=float, default=10.0)
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()
#     logging.basicConfig(level=logging.INFO, format="%(message)s")
#     client = WikipediaClient(timeout=args.timeout)
#     result = navigate(
#         args.start_page,
#         args.target_page,
#         client=client,
#         max_steps=args.max_steps,
#         link_limit=args.link_limit,
#     )

#     for log in result.logs:
#         print(
#             f"Visited {log.page!r}: {log.links_found} links in "
#             f"{log.elapsed_seconds:.3f}s -> {log.selected_link!r} "
#             f"(score={log.selected_score})"
#         )
#     print("Path:", " -> ".join(result.path))
#     print(f"Reached target: {result.reached_target} ({result.reason})")

#load nn config from file, pretrained

def speedrun(start_page, end_page, config):
    wiki = WikipediaClient()
    page_history = []
    t_init = time.perf_counter()
    time_history = [t_init]

    t_latest = time.perf_counter()

    print(f"Start page: {start_page}. End page: {end_page}.")
    current_page = start_page
    page_history.append(current_page)
    #pages_visited = 0
    while current_page != end_page:
        #identify closest hyperlink
        closest_hyperlink = find_closest_hyperlink(current_page, end_page, config, wiki)

        
        t_latest = time.perf_counter()
        print(f"Visiting page: {closest_hyperlink}. Time: {t_latest - t_init} \n")
        time_history.append(t_latest)
        page_history.append(closest_hyperlink)

        #set current page equal to closest hyperlink
        current_page = closest_hyperlink

        #pages_visited += 1

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

def find_closest_hyperlink(current_page, end_page, config, wiki_client : WikipediaClient):
    hyperlinks = wiki_client.get_hyperlinks(current_page) #TODO: insert from wikipedia client

    #get word ids
    max_similarity = -1
    closest_page = None
    for page in hyperlinks:
        if page == current_page:
            continue
        similarity = compare_titles(page, end_page, config['word_to_id'], config['model'])        
        if similarity > max_similarity:
            max_similarity = similarity
            closest_page = page
    return closest_page
    
def main():
    start_page = START_PAGE
    end_page = END_PAGE

    model, word_to_id, id_to_word, model_config, ckpt = load_checkpoint(
        LOAD_MODEL_PATH,
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