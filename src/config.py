from pathlib import Path

#embedding structure
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
PADDING_IDX = 0
UNKNOWN_IDX = 1
MAX_SEQ_LENGTH = 500
MIN_FREQ = 1
WINDOW = 5
MAX_LINES = 10_000  #max lines to read from corpus
NEG_K = 3

#training parameters
LEARNING_RATE = 0.0015
BATCH_SIZE = 256
EPOCHS = 10

#path
DATASET_PATH = "emmermarcell/wikipedia-corpus-2023-03-01"
USE_FILES = ["wikipedia_processed_4.txt"]
CACHE_DIR = Path("./libs/dataset_cache")
EXTRACT_DIR = CACHE_DIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = "./runs/sgns_wiki_v5"

LOAD_MODEL_PATH = "runs/sgns_wiki_v5"

START_PAGE = "Python (Programming Language)"
END_PAGE = "Cognition"

LINK_LIMIT = 1000

# parsed Wikipedia dataset
WIKIPEDIA_JSONL_PATH = Path(
    "data/processed/wikipedia_test.jsonl"
)