import numpy as np
import torch
from src.page_reader import SGNSDataset

def test_streaming_dataset_returns_expected_shapes():
    # tokenized_docs: list of lists (include PAD=0)
    np.random.seed(0)
    torch.manual_seed(0)

    tokenized_docs = [
        [2, 3, 4, 0, 0],
        [5, 6, 0, 0, 0],
    ]
    vocab_size = 10
    probs = np.ones(vocab_size) / vocab_size
    probs[0] = 0  # pad
    probs[1] = 0  # unk
    probs /= probs.sum()

    ds = SGNSDataset(tokenized_docs, probs, neg_k=4, window=2, padding_idx=0)

    center, context, neg = ds[0]
    assert center.shape == torch.Size([])
    assert context.shape == torch.Size([])
    assert neg.shape == torch.Size([4])

    assert center.item() != 0
    assert context.item() != 0
    assert all(n.item() >= 0 for n in neg)

def test_streaming_dataset_len_is_total_nonpad_tokens():
    tokenized_docs = [
        [2, 3, 4, 0, 0],  # 3 non-pad
        [5, 6, 0, 0, 0],  # 2 non-pad
    ]
    vocab_size = 10
    probs = np.ones(vocab_size) / vocab_size
    probs[0] = 0
    probs[1] = 0
    probs /= probs.sum()

    ds = SGNSDataset(tokenized_docs, probs, neg_k=2, window=2, padding_idx=0)
    assert len(ds) == 5
