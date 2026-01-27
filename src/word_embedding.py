import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import time

from src.page_reader import SGNSDataset, load_dataset, build_sgns_dataloader, merge_vocabularies

from src.save_configs import load_checkpoint, save_checkpoint

#hyperparameters
from src.constants import MAX_LINES, LOAD_MODEL_PATH, SAVE_DIR, WINDOW, MIN_FREQ, MAX_SEQ_LENGTH, USE_FILES, VOCAB_SIZE, EMBEDDING_DIM, PADDING_IDX, EPOCHS, LEARNING_RATE, BATCH_SIZE, DATASET_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="Skip-gram with Negative Sampling Word Embedding Model")
    #parser.add_argument('--vocab_size', type=int, default=VOCAB_SIZE, help='Size of the vocabulary')
    parser.add_argument('--embedding_dim', type=int, default=EMBEDDING_DIM, help='Dimension of the word embeddings')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--neg_samples', type=int, default=5, help='Number of negative samples per positive sample')
    parser.add_argument('--seed', type = int, default=0)
    parser.add_argument('--load_model', action="store_true", help='Use path to a pre-trained model checkpoint to load')
    parser.add_argument('--max_lines', type=int, default=MAX_LINES, help='Maximum number of lines to read from the dataset')
    return parser.parse_args()

class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM):
        super(SkipGramNegSampling, self).__init__()
        self.in_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_IDX, sparse=True)
        self.out_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_IDX, sparse=True)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.in_embedding.weight, -0.5, 0.5)
        nn.init.zeros_(self.out_embedding.weight)
    
    def forward(self, center_ids, pos_ids, neg_ids):
        u = self.in_embedding(center_ids)
        v_pos = self.out_embedding(pos_ids)
        v_neg = self.out_embedding(neg_ids)

        pos_logits = (u * v_pos).sum(dim=1)
        neg_logits = torch.bmm(v_neg, u.unsqueeze(2)).squeeze(2)

        loss = -(F.logsigmoid(pos_logits) + F.logsigmoid(-neg_logits).sum(dim=1)).mean()

        return loss
    
    def fit(self, dataloader, device, optimizer, total_epochs=EPOCHS, log_every=1_000):
        self.to(device)
        t0 = time.time()
        start = t0
        for epoch in range(1, total_epochs + 1):
            t0 = time.perf_counter()
            last_idx  = 0
            self.train()
            total_loss = 0
            for batch_idx, (center_words, context_words, negative_words) in enumerate(dataloader):
                center_words = center_words.to(device, non_blocking=True)
                context_words = context_words.to(device, non_blocking=True)
                negative_words = negative_words.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                loss = self(center_words, context_words, negative_words)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % log_every == 0:
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    avg = total_loss / (batch_idx + 1)
                    t1 = time.perf_counter()
                    elapsed = t1 - t0
                    batches_per_sec = (batch_idx - last_idx) / elapsed
                    last_idx = batch_idx
                    t0 = t1
                    print(
                        f"[Epoch {epoch} / {total_epochs}] "
                        f"Batch {batch_idx}/{len(dataloader)} | "
                        f"Loss={avg:.4f} | "
                        f"{batches_per_sec:.2f} batches/s"
                    )

            print(f"Epoch {epoch} / {total_epochs}: loss={avg:.4f}, time={time.time()-start:.1f}s")

    @staticmethod
    def expand_embeddings_inplace(model, new_vocab_size, init_std=0.01):
        old_vocab_size, emb_dim = model.in_embedding.weight.shape
        if new_vocab_size <= old_vocab_size:
            return

        device = model.in_embedding.weight.device

        new_in = nn.Embedding(new_vocab_size, emb_dim, padding_idx=PADDING_IDX, sparse=True).to(device)
        new_out = nn.Embedding(new_vocab_size, emb_dim, padding_idx=PADDING_IDX, sparse=True).to(device)

        with torch.no_grad():
            new_in.weight[:old_vocab_size].copy_(model.in_embedding.weight)
            new_out.weight[:old_vocab_size].copy_(model.out_embedding.weight)

            nn.init.normal_(new_in.weight[old_vocab_size:], mean=0.0, std=init_std)
            nn.init.zeros_(new_out.weight[old_vocab_size:])

        model.in_embedding = new_in
        model.out_embedding = new_out

def expand_embeddings_inplace(model, new_vocab_size, init_std=0.01):
    old_vocab_size, emb_dim = model.in_embedding.weight.shape
    if new_vocab_size <= old_vocab_size:
        return

    device = model.in_embedding.weight.device

    new_in = nn.Embedding(new_vocab_size, emb_dim, padding_idx=PADDING_IDX, sparse=True).to(device)
    new_out = nn.Embedding(new_vocab_size, emb_dim, padding_idx=PADDING_IDX, sparse=True).to(device)

    with torch.no_grad():
        new_in.weight[:old_vocab_size].copy_(model.in_embedding.weight)
        new_out.weight[:old_vocab_size].copy_(model.out_embedding.weight)

        nn.init.normal_(new_in.weight[old_vocab_size:], mean=0.0, std=init_std)
        nn.init.zeros_(new_out.weight[old_vocab_size:])

    model.in_embedding = new_in
    model.out_embedding = new_out

def check_batch_vs_model(dataloader, model, device):
    model_vocab = model.in_embedding.num_embeddings
    batch = next(iter(dataloader))
    centers, contexts, neg = batch

    max_id = max(
        int(centers.max()),
        int(contexts.max()),
        int(neg.max())
    )

    print(f"[DEBUG] model_vocab={model_vocab:,}  max_batch_id={max_id:,}")
    if max_id >= model_vocab:
        raise ValueError(f"Batch has id {max_id} but model vocab is {model_vocab}")




def main():
    args = parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")

    samples, dataloader, word_to_id, id_to_word = load_dataset(max_lines=args.max_lines)

    torch.manual_seed(args.seed)
    if (args.load_model is True) and (LOAD_MODEL_PATH is not None):
        print(f"[INFO] Loading model from {LOAD_MODEL_PATH}...")
        model, word_to_id_ckpt, id_to_word_ckpt, config, ckpt = load_checkpoint(LOAD_MODEL_PATH, SkipGramNegSampling, map_location=device)
        merged_word_to_id, merged_id_to_word = merge_vocabularies(word_to_id_ckpt, word_to_id, id_to_word_ckpt, id_to_word)
        dataloader, word_to_id, id_to_word = build_sgns_dataloader(samples, vocab_override=(merged_word_to_id, merged_id_to_word))
        expanded_vocab_size = len(word_to_id)
        expand_embeddings_inplace(model, expanded_vocab_size)
        print("[DONE] Model Loaded.")
    else:
        model = SkipGramNegSampling(vocab_size=len(word_to_id), embedding_dim=args.embedding_dim).to(device)
    optimizer = optim.SparseAdam(model.parameters(), lr=args.learning_rate)
    print("[DEBUG] len(word_to_id)=", len(word_to_id))
    print("[DEBUG] model_vocab=", model.in_embedding.num_embeddings)

    check_batch_vs_model(dataloader, model, device)
    model.fit(dataloader, device, optimizer=optimizer, total_epochs=args.epochs)
    
    config = {
        "vocab_size": len(word_to_id),
        "embedding_dim": args.embedding_dim,
        "padding_idx": PADDING_IDX,
        "dataset": DATASET_PATH,
        "use_files": USE_FILES,
        "window": WINDOW,
        "neg_k": args.neg_samples,
        "max_seq_length": MAX_SEQ_LENGTH,
        "min_freq": MIN_FREQ,
    }

    save_checkpoint(
        save_dir=SAVE_DIR,
        model=model,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        config=config,
        optimizer=optimizer,
        total_epochs=args.epochs,
        last_loss=None,
    )

if __name__ == "__main__":
    main()
