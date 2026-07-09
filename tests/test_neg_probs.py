import numpy as np
from src.training.page_reader import neg_probs, build_vocabulary, PADDING_TOKEN, UNKNOWN_TOKEN
from src.config import PADDING_IDX, UNKNOWN_IDX

def test_neg_probs_normalizes_and_excludes_special_tokens():
    
    corpus = ["dog cat dog", "cat mouse"]
    word_to_id, id_to_word, counts = build_vocabulary(corpus, min_freq=1)
    probs = neg_probs(counts, word_to_id)

    assert np.isclose(probs.sum(), 1.0)
    assert probs[PADDING_IDX] == 0.0
    assert probs[UNKNOWN_IDX] == 0.0