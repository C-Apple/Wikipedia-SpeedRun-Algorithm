from src.page_reader import build_vocabulary, PADDING_TOKEN, UNKNOWN_TOKEN
from src.constants import PADDING_IDX, UNKNOWN_IDX

def test_build_vocabulary_includes_special_tokens():
    corpus = ["Hello world", "hello there"]
    word_to_id, id_to_word, counts = build_vocabulary(corpus, min_freq=1)

    assert word_to_id[PADDING_TOKEN] == PADDING_IDX
    assert word_to_id[UNKNOWN_TOKEN] == UNKNOWN_IDX
    assert id_to_word[PADDING_IDX] == PADDING_TOKEN
    assert id_to_word[UNKNOWN_IDX] == UNKNOWN_TOKEN

def test_build_vocabulary_min_freq_filters():
    corpus = ["a b b", "c"]
    word_to_id, id_to_word, counts = build_vocabulary(corpus, min_freq=2)
    assert "b" in word_to_id
    assert "a" not in word_to_id
    assert "c" not in word_to_id
