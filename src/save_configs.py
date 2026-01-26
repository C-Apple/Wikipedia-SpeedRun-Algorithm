from pathlib import Path
import json
import torch

def save_checkpoint(
    save_dir: str | Path,
    model,
    word_to_id: dict,
    id_to_word: dict,
    config: dict,
    optimizer=None,
    total_epochs: int | None = None,
    last_loss: float | None = None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save model weights (+ optimizer if provided)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "epochs": total_epochs,
        "last_loss": last_loss,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(ckpt, save_dir / "checkpoint.pt")

    # 2) Save vocab as JSON (human-readable)
    with open(save_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {"word_to_id": word_to_id, "id_to_word": {str(k): v for k, v in id_to_word.items()}},
            f
        )

    print(f"Saved checkpoint to: {save_dir}")

def load_checkpoint(load_dir: str | Path, model_class):
    load_dir = Path(load_dir)

    ckpt = torch.load(load_dir / "checkpoint.pt", map_location="cpu")
    with open(load_dir / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    word_to_id = vocab["word_to_id"]
    id_to_word = {int(k): v for k, v in vocab["id_to_word"].items()}

    config = ckpt["config"]

    # Recreate model with saved config
    model = model_class(vocab_size=config["vocab_size"], embedding_dim=config["embedding_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, word_to_id, id_to_word, config, ckpt
