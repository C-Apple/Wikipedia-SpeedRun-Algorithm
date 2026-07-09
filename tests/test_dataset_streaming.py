import numpy as np
import torch

from src.training.page_reader import SGNSDataset, SGNSCollator


def test_streaming_dataset_returns_expected_types_and_ranges():
    np.random.seed(0)
    torch.manual_seed(0)

    tokenized_docs = [
        [2, 3, 4, 0, 0],
        [5, 6, 0, 0, 0],
    ]

    vocab_size = 10
    probs = np.ones(vocab_size, dtype=np.float32)
    probs[0] = 0  # pad
    probs[1] = 0  # unk
    probs /= probs.sum()

    ds = SGNSDataset(tokenized_docs, probs, neg_k=4, window=2, padding_idx=0)

    center, context = ds[0]
    assert isinstance(center, int)
    assert isinstance(context, int)
    assert center != 0
    assert context != 0
    assert 0 <= center < vocab_size
    assert 0 <= context < vocab_size


def test_collator_outputs_expected_shapes_and_ranges():
    torch.manual_seed(0)

    vocab_size = 10
    probs = torch.ones(vocab_size, dtype=torch.float32)
    probs[0] = 0
    probs[1] = 0
    probs = probs / probs.sum()

    K = 4
    collator = SGNSCollator(neg_probs_t=probs, K=K)

    # Make a fake batch of (center, context) ints like the dataset returns
    batch = [(2, 3), (5, 6), (4, 2)]
    centers, contexts, neg = collator(batch)

    assert centers.shape == (len(batch),)
    assert contexts.shape == (len(batch),)
    assert neg.shape == (len(batch), K)

    # PAD/UNK should never appear because prob mass is zero
    assert not (neg == 0).any()
    assert not (neg == 1).any()

    assert (neg >= 0).all()
    assert (neg < vocab_size).all()


def test_streaming_dataset_len_is_total_nonpad_tokens():
    tokenized_docs = [
        [2, 3, 4, 0, 0],  # 3 non-pad
        [5, 6, 0, 0, 0],  # 2 non-pad
    ]

    vocab_size = 10
    probs = np.ones(vocab_size, dtype=np.float32)
    probs[0] = 0
    probs[1] = 0
    probs /= probs.sum()

    ds = SGNSDataset(tokenized_docs, probs, neg_k=2, window=2, padding_idx=0)
    assert len(ds) == 5
