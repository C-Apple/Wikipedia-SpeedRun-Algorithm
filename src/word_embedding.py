import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import time

from src.page_reader import SGNSDataset, load_dataset, build_sgns_dataloader

from src.save_configs import save_checkpoint

#hyperparameters
from src.constants import WINDOW, MIN_FREQ, MAX_SEQ_LENGTH, USE_FILES, VOCAB_SIZE, EMBEDDING_DIM, PADDING_IDX, EPOCHS, LEARNING_RATE, BATCH_SIZE, DATASET_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="Skip-gram with Negative Sampling Word Embedding Model")
    #parser.add_argument('--vocab_size', type=int, default=VOCAB_SIZE, help='Size of the vocabulary')
    parser.add_argument('--embedding_dim', type=int, default=EMBEDDING_DIM, help='Dimension of the word embeddings')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--neg_samples', type=int, default=5, help='Number of negative samples per positive sample')
    parser.add_argument('--seed', type = int, default=0)
    return parser.parse_args()
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM):
        super(SkipGramNegSampling, self).__init__()
        self.in_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_IDX)
        self.out_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_IDX)
        
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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, dataloader, device, optimizer, total_epochs=EPOCHS, log_every=100):
        self.to(device)
        start = time.time()
        for epoch in range(1, total_epochs + 1):
            self.train()
            total_loss = 0
            for batch_idx, (center_words, context_words, negative_words) in enumerate(dataloader):
                center_words = center_words.to(device)
                context_words = context_words.to(device)
                negative_words = negative_words.to(device)

                optimizer.zero_grad(set_to_none=True)
                loss = self(center_words, context_words, negative_words)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % log_every == 0:
                    avg = total_loss / (batch_idx + 1)
                    elapsed = time.time() - start
                    batches_per_sec = batch_idx / elapsed

                    print(
                        f"[Epoch {epoch} / {total_epochs}] "
                        f"Batch {batch_idx}/{len(dataloader)} | "
                        f"Loss={avg:.4f} | "
                        f"{batches_per_sec:.2f} batches/s"
                    )
            print(f"Epoch {epoch} / {total_epochs}: loss={avg:.4f}, time={time.time()-start:.1f}s")
    
def main():
    args = parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")

    dataloader, word_to_id, id_to_word = load_dataset()

    torch.manual_seed(args.seed)

    model = SkipGramNegSampling(vocab_size=len(word_to_id), embedding_dim=args.embedding_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
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
        save_dir="./runs/sgns_wiki",
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
