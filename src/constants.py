from pathlib import Path

#embedding structure
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
PADDING_IDX = 0
UNKNOWN_IDX = 1
MAX_SEQ_LENGTH = 500
MIN_FREQ = 2
WINDOW = 5
MAX_LINES = 50_000  #max lines to read from corpus
NEG_K = 3

#training parameters
LEARNING_RATE = 0.0015
BATCH_SIZE = 512
EPOCHS = 1

#path
DATASET_PATH = "emmermarcell/wikipedia-corpus-2023-03-01"
USE_FILES = ["wikipedia_processed_1.txt"]
CACHE_DIR = Path("./dataset_cache")
EXTRACT_DIR = CACHE_DIR / "extracted"
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)