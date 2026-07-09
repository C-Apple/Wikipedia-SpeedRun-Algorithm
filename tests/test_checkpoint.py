import torch
import torch.nn as nn
from pathlib import Path
from src.training.save_configs import save_checkpoint, load_checkpoint

class TinyModel(nn.Module):
    def __init__(self, vocab_size=7, embedding_dim=5):
        super().__init__()
        self.in_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.out_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

def test_save_and_load_checkpoint(tmp_path: Path):
    model = TinyModel(vocab_size=7, embedding_dim=5)
    word_to_id = {"<PAD>": 0, "<UNK>": 1, "a": 2}
    id_to_word = {0: "<PAD>", 1: "<UNK>", 2: "a"}
    config = {"vocab_size": 7, "embedding_dim": 5}

    save_checkpoint(tmp_path, model, word_to_id, id_to_word, config)

    loaded_model, w2i, i2w, cfg, ckpt = load_checkpoint(tmp_path, TinyModel)

    assert cfg["vocab_size"] == 7
    assert cfg["embedding_dim"] == 5
    assert w2i["a"] == 2
    assert i2w[2] == "a"

    # weights match
    for k, v in model.state_dict().items():
        assert torch.allclose(v, loaded_model.state_dict()[k])
