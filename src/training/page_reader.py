from collections import Counter
import torch
import json
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import zipfile
import bisect
import re

#hyperparameters
from src.config import WIKIPEDIA_JSONL_PATH, WINDOW, MAX_LINES, MIN_FREQ, PADDING_IDX, UNKNOWN_IDX, MAX_SEQ_LENGTH, BATCH_SIZE, DATASET_PATH, CACHE_DIR, EXTRACT_DIR, USE_FILES, NEG_K

PADDING_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
#insert page reading later

ENTITY_PREFIX = "wiki::"


def is_entity_token(token: str) -> bool:
    return token.startswith(ENTITY_PREFIX)

def neg_probs(counts, word_to_id, power=0.75):
    vocab_size = sum(counts.values())
    probs = np.zeros(len(word_to_id))

    if vocab_size == 0:
        probs[:] = 1.0 / len(word_to_id)
        probs[PADDING_IDX] = 0
        probs[UNKNOWN_IDX] = 0
        probs /= probs.sum()
        return probs
    
    for word, idx in word_to_id.items():
        if word in [PADDING_TOKEN, UNKNOWN_TOKEN]:
            probs[idx] = 0.0
            continue
        freq = counts.get(word, 0)
        probs[idx] = (freq / vocab_size) ** power

    s = probs.sum()
    if s == 0:
        probs[:] = 1.0 / len(word_to_id)
        probs[PADDING_IDX] = 0
        probs[UNKNOWN_IDX] = 0
        probs /= probs.sum()
        return probs
    probs /= s
    return probs

def pad_sequence(sequence, max_length=MAX_SEQ_LENGTH, padding_value=PADDING_IDX):
    if len(sequence) >= max_length:
        return sequence[:max_length]
    else:
        return sequence + [padding_value] * (max_length - len(sequence))

def tokenize_string(input_string):
    return TOKEN_RE.findall(input_string.lower())

class SGNSDataset(Dataset):
    def __init__(self, tokenized_docs, neg_probs, neg_k, window=WINDOW, padding_idx=PADDING_IDX):
        # 1) remove PAD from each doc
        self.docs = [[t for t in doc if t != padding_idx] for doc in tokenized_docs]
        # 2) drop docs that are too short to form pairs
        self.docs = [doc for doc in self.docs if len(doc) >= 2]
        self.neg_probs = neg_probs
        self.window = window
        self.neg_k = neg_k
        self.padding_idx = padding_idx
        self.vocab_size = len(neg_probs)

        lengths = [len(d) for d in self.docs]
        self.cum = np.cumsum(lengths)  # e.g. [5, 8, 12, ...]
        self.total_tokens = int(self.cum[-1]) if len(self.cum) else 0

    def __len__(self):
        return self.total_tokens
    
    def __getitem__(self, idx):
        # Map global idx -> (doc_i, pos_i)
        #if idx == 0:
            #print("[DEBUG] Getting item at idx=0")
        doc_i = bisect.bisect_right(self.cum, idx)
        prev_cum = int(self.cum[doc_i - 1]) if doc_i > 0 else 0
        pos_i = int(idx - prev_cum)

        doc = self.docs[doc_i]
        center = doc[pos_i]

        # Sample one positive context within window
        left = max(0, pos_i - self.window)
        right = min(len(doc), pos_i + self.window + 1)

        left_num = pos_i - left
        right_num = right - pos_i - 1
        total_contexts = left_num + right_num

        r = np.random.randint(total_contexts)
        if r < left_num:
            ctx_pos = pos_i - (r + 1)
        else:
            ctx_pos = pos_i + (r - left_num + 1)

        context = doc[ctx_pos]

        return (
            int(center), int(context)
        )

def build_vocabulary(tokenized_docs, min_freq=MIN_FREQ, entity_min_freq=2):
    counter = Counter()

    word_to_id = {
        PADDING_TOKEN: PADDING_IDX,
        UNKNOWN_TOKEN: UNKNOWN_IDX,
    }

    id_to_word = {
        PADDING_IDX: PADDING_TOKEN,
        UNKNOWN_IDX: UNKNOWN_TOKEN,
    }

    for tokens in tokenized_docs:
        counter.update(tokens)

    for token, freq in sorted(
        counter.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        required_freq = (
            entity_min_freq
            if is_entity_token(token)
            else min_freq
        )

        if freq < required_freq:
            continue

        if token not in word_to_id:
            idx = len(word_to_id)
            word_to_id[token] = idx
            id_to_word[idx] = token

    return word_to_id, id_to_word, counter

def numericalize_corpus(tokenized_docs, word_to_id):
    numericalized_docs = []

    for tokens in tokenized_docs:
        token_ids = [
            word_to_id.get(token, UNKNOWN_IDX)
            for token in tokens
        ]

        padded_ids = pad_sequence(token_ids)
        numericalized_docs.append(padded_ids)

        if len(numericalized_docs) % 100_000 == 0:
            print(
                f"[INFO] Numericalized "
                f"{len(numericalized_docs):,} documents"
            )

    return numericalized_docs

def vocab_tensorize(document, word_to_id):
    if isinstance(document, str):
        tokens = tokenize_string(document)
    else:
        tokens = document
    
    token_ids = [
        word_to_id.get(token, UNKNOWN_IDX)
        for token in tokens
    ]

    padded_ids = pad_sequence(token_ids)

    return torch.tensor(
        padded_ids,
        dtype=torch.long,
    )

def load_wikipedia_jsonl(
    path,
    max_articles=MAX_LINES,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find parsed Wikipedia file: {path}"
        )

    tokenized_docs = []
    article_titles = []
    article_links = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

            tokens = record.get("tokens")

            if not isinstance(tokens, list):
                raise ValueError(
                    f"Article on line {line_number} "
                    f"does not contain a valid 'tokens' list. "
                    f"Regenerate the JSONL with token extraction enabled."
                )

            tokens = [
                token
                for token in tokens
                if isinstance(token, str) and token
            ]

            if len(tokens) < 2:
                continue

            for chunk in chunk_tokens(tokens):
                tokenized_docs.append(chunk)
            article_titles.append(record.get("title"))
            article_links.append(record.get("links", []))

            if len(tokenized_docs) % 10_000 == 0:
                print(
                    f"[INFO] Loaded "
                    f"{len(tokenized_docs):,} articles"
                )

            if (
                max_articles is not None
                and len(tokenized_docs) >= max_articles
            ):
                break

    print(
        f"[DONE] Loaded "
        f"{len(tokenized_docs):,} tokenized articles"
    )

    return tokenized_docs, article_titles, article_links

def save_vocab(path, word_to_id, max_length, padding_idx=PADDING_IDX, unk_idx=UNKNOWN_IDX, min_freq=2):
    payload = {
        "word_to_id": word_to_id,
        "max_length": max_length,
        "padding_idx": padding_idx,
        "unk_idx": unk_idx,
        "min_freq": min_freq,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload

def make_dataloader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=None):
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else 0
    collate_fn = collate_fn

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor, collate_fn=collate_fn)

def counts_from_corpus(tokenized_docs, word_to_id):
    counts = Counter()

    for tokens in tokenized_docs:
        for token in tokens:
            if (
                token in word_to_id
                and token not in (
                    PADDING_TOKEN,
                    UNKNOWN_TOKEN,
                )
            ):
                counts[token] += 1

    return counts


def build_sgns_dataloader(tokenized_docs, vocab_override=None, window=WINDOW, neg_k=NEG_K, min_freq=MIN_FREQ, batch_size=BATCH_SIZE, shuffle=True, num_workers=4):
    #1) Build vocab
    if vocab_override is not None:
        word_to_id, id_to_word = vocab_override
        counts = counts_from_corpus(tokenized_docs, word_to_id)
        print(f"[INFO] Using overridden vocabulary of size {len(word_to_id):,}.")
    else:
        print("[INFO] Building Vocabulary...")
        word_to_id, id_to_word, counts = build_vocabulary(tokenized_docs, min_freq)
        print(f"[DONE] Vocab Size: {len(word_to_id):,}.")
    #2) Numericalize tokenized_docs
    print("[INFO] Tokenizing Corpus...")
    tokenized_doc = numericalize_corpus(tokenized_docs, word_to_id)
    print("[DONE] Tokenization Complete")
    #3) Generate (center, context) pairs
    #pairs = generate_pairs(tokenized_doc.flatten().tolist(), window=window)

    #4) sampling neg distribution
    print("[INFO] Computing Negative Sampling Probabilities...")
    neg_probabilities = neg_probs(counts, word_to_id)
    print("[DONE] Negative Sampling Probabilities Computed")
    neg_probs_t = torch.tensor(neg_probabilities, dtype=torch.float32)
    #5) Create Dataset and DataLoader
    print("[INFO] Creating Collator...")
    collator = SGNSCollator(neg_probs_t, neg_k)
    print("[DONE] Collator Created")

    print("[INFO] Creating DataLoader...")
    dataset = SGNSDataset(tokenized_doc, neg_probabilities, neg_k, window=window)
    dataloader = make_dataloader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collator)
    print("[DONE] DataLoader Created")

    center, context, neg = next(iter(dataloader))
    print(center.shape, context.shape, neg.shape)

    return dataloader, word_to_id, id_to_word

def has_any_files(d):
    return any(p.is_file() for p in d.rglob("*"))

def merge_vocabularies(first_word_to_id, second_word_to_id, first_id_to_word, second_id_to_word):
    merged_word_to_id = dict(first_word_to_id)
    merged_id_to_word = dict(first_id_to_word)

    for word, idx in second_word_to_id.items():
        if word not in merged_word_to_id:
            new_idx = len(merged_word_to_id)
            merged_word_to_id[word] = new_idx
            merged_id_to_word[new_idx] = word

    return merged_word_to_id, merged_id_to_word

def chunk_tokens(tokens, chunk_size=MAX_SEQ_LENGTH):
    return [
        tokens[start:start + chunk_size]
        for start in range(0, len(tokens), chunk_size)
        if len(tokens[start:start + chunk_size]) >= 2
    ]

#download dataset from kaggle
def load_dataset(
    dataset_path=WIKIPEDIA_JSONL_PATH,
    max_lines=MAX_LINES,
    neg_k=NEG_K,
):
    tokenized_docs, article_titles, article_links = (
        load_wikipedia_jsonl(
            dataset_path,
            max_articles=max_lines,
        )
    )

    dataloader, word_to_id, id_to_word = (
        build_sgns_dataloader(
            tokenized_docs,
            neg_k=neg_k,
        )
    )

    return (
        tokenized_docs,
        dataloader,
        word_to_id,
        id_to_word,
    )
class SGNSCollator:
    def __init__(self, neg_probs_t: torch.Tensor, K: int):
        self.neg_probs_t = neg_probs_t
        self.K = K

    def __call__(self, batch):
        centers, contexts = zip(*batch)
        centers = torch.tensor(centers, dtype=torch.long)
        contexts = torch.tensor(contexts, dtype=torch.long)

        B = centers.size(0)
        neg = torch.multinomial(self.neg_probs_t, num_samples=B*self.K, replacement=True).view(B, self.K)
        return centers, contexts, neg