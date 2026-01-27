from pathlib import Path

#embedding structure
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
PADDING_IDX = 0
UNKNOWN_IDX = 1
MAX_SEQ_LENGTH = 500
MIN_FREQ = 2
WINDOW = 5
MAX_LINES = 10_000  #max lines to read from corpus
NEG_K = 3

#training parameters
LEARNING_RATE = 0.0015
BATCH_SIZE = 512
EPOCHS = 8

#path
DATASET_PATH = "emmermarcell/wikipedia-corpus-2023-03-01"
USE_FILES = ["wikipedia_processed_4.txt"]
CACHE_DIR = Path("./libs/dataset_cache")
EXTRACT_DIR = CACHE_DIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = "./runs/sgns_wiki_v4"

LOAD_MODEL_PATH = "runs/sgns_wiki_v2"

START_PAGE = "Python (programming language)"
END_PAGE = "Artificial intelligence"