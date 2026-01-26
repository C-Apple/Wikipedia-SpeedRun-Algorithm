from collections import Counter
import torch
import json
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import kagglehub
from pathlib import Path
import zipfile
import bisect



#hyperparameters
from src.constants import WINDOW, MAX_LINES, MIN_FREQ, PADDING_IDX, UNKNOWN_IDX, MAX_SEQ_LENGTH, BATCH_SIZE, DATASET_PATH, CACHE_DIR, EXTRACT_DIR, USE_FILES

PADDING_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'
#insert page reading later

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
    return input_string.lower().split()

def build_vocabulary(corpus, min_freq=2):
    counter = Counter()
    word_to_id = {PADDING_TOKEN: PADDING_IDX, UNKNOWN_TOKEN: UNKNOWN_IDX} #padding idx = 0
    id_to_word = {PADDING_IDX: PADDING_TOKEN, UNKNOWN_IDX: UNKNOWN_TOKEN}
    for document in corpus:
        tokens = tokenize_string(document)
        counter.update(tokens)

    for word, freq in counter.items():
        if freq >= min_freq and word not in word_to_id:
            idx = word_to_id.__len__()
            word_to_id[word] = idx
            id_to_word[idx] = word
    
    return word_to_id, id_to_word, counter

def numericalize_corpus(corpus, word_to_id):
    tokenized_doc = []
    for document in corpus:
        tokens = tokenize_string(document)
        token_ids = [word_to_id.get(token, UNKNOWN_IDX) for token in tokens]
        padded_ids = pad_sequence(token_ids)
        tokenized_doc.append(padded_ids)
    return tokenized_doc

# def generate_pairs(token_ids, window):
#     pairs = []
#     n = len(token_ids)
#     for i in range(n):
#         center = token_ids[i]
#         left = max(0, i-window)
#         right = min(n, i+window+1)
#         for j in range(left, right):
#             if j == i:
#                 continue
#             pos = token_ids[j]
#             pairs.append((center, pos))
#     return pairs

def vocab_tensorize(document, word_to_id):
    tokens = tokenize_string(document)
    token_ids = [word_to_id.get(token, UNKNOWN_IDX) for token in tokens]
    padded_ids = pad_sequence(token_ids)
    return torch.tensor(padded_ids, dtype=torch.long)

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
        doc_i = bisect.bisect_right(self.cum, idx)
        prev_cum = int(self.cum[doc_i - 1]) if doc_i > 0 else 0
        pos_i = int(idx - prev_cum)

        doc = self.docs[doc_i]
        center = doc[pos_i]

        # Sample one positive context within window
        left = max(0, pos_i - self.window)
        right = min(len(doc), pos_i + self.window + 1)

        candidates = list(range(left, pos_i)) + list(range(pos_i + 1, right))
        ctx_pos = candidates[np.random.randint(len(candidates))]
        context = doc[ctx_pos]

        # Sample negatives
        neg = np.random.choice(self.vocab_size, size=self.neg_k, p=self.neg_probs)

        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )
    
def make_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def build_sgns_dataloader(corpus, window=MIN_FREQ, neg_k=5, min_freq=MIN_FREQ, batch_size=BATCH_SIZE, shuffle=True, num_workers=0):
    #1) Build vocab
    word_to_id, id_to_word, counts = build_vocabulary(corpus, min_freq)
    print("Vocab Size:", len(word_to_id))
    #2) Numericalize corpus
    tokenized_doc = numericalize_corpus(corpus, word_to_id)

    #3) Generate (center, context) pairs
    #pairs = generate_pairs(tokenized_doc.flatten().tolist(), window=window)

    #4) sampling neg distribution
    neg_probabilities = neg_probs(counts, word_to_id)

    #5) Create Dataset and DataLoader
    dataset = SGNSDataset(tokenized_doc, neg_probabilities, neg_k, window=window)
    dataloader = make_dataloader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    center, context, neg = next(iter(dataloader))
    print(center.shape, context.shape, neg.shape)

    return dataloader, word_to_id, id_to_word


def has_any_files(d):
    return any(p.is_file() for p in d.rglob("*"))

#download dataset from kaggle
def load_dataset(dataset_path=DATASET_PATH, extract_dir=EXTRACT_DIR, use_files=USE_FILES, max_lines=MAX_LINES):
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    download_path = Path(kagglehub.dataset_download(dataset_path))
    print("Downloaded to:", download_path)
    
    flag = extract_dir / "extracted.flag"
    if flag.exists() and not has_any_files(extract_dir):
        flag.unlink()

    if (not flag.exists() or not has_any_files(extract_dir)):
            zips = list(download_path.glob("*.zip")) if download_path.is_dir() else []
            if download_path.is_file() and download_path.suffix == ".zip":
                zips = [download_path]

            for z in zips:
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall(extract_dir)

            flag.write_text("ok")
            print("Unzipped to:", extract_dir)
    else:
        print("Already unzipped:", extract_dir)
        print("Top-level in extract_dir:")
        if extract_dir.exists():
            for x in sorted(extract_dir.iterdir())[:50]:
                print(" -", x.name)

        ext_counts = Counter(p.suffix.lower() for p in extract_dir.rglob("*") if p.is_file())
        print("Extensions in extract_dir:", ext_counts)

    samples = []
    root_dir = Path(download_path)
    print("Using root dir: ", root_dir)

    for name in use_files:
        matches = list(root_dir.rglob(name))
        if not matches:
            # helpful debug: list a few txt files that DO exist
            txts = list(root_dir.rglob("*.txt"))
            sample = "\n".join(str(t) for t in txts[:30])
            raise FileNotFoundError(
                f"Could not find {name} under {root_dir}.\n"
                f"Here are some .txt files I *did* find (first 30):\n{sample}"
            )

        p = matches[0]  # take first match

        print("Reading:", p)
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                samples.append(line)

                # progress heartbeat
                if len(samples) % 1000 == 0:
                    print(f"[INFO] Collected {len(samples):,} lines so far")

                # early exit
                if max_lines is not None and len(samples) >= max_lines:
                    print(f"[DONE] Reached max_lines = {max_lines:,}")
                    break

    dataloader, word_to_id, id_to_word = build_sgns_dataloader(samples)
    
    return dataloader, word_to_id, id_to_word