# Wikipedia SpeedRun Algorithm

This project explores whether machine learning can be used to solve Wikipedia Speedruns.

The idea started with a simple question: if two Wikipedia articles are semantically similar, can that information help navigate from one article to another using only hyperlinks?

Instead of using a traditional shortest-path algorithm, I wanted to see if a model trained on Wikipedia itself could develop an intuition for where to go next.

To do that, I implemented a Skip-Gram with Negative Sampling (SGNS) model in PyTorch and trained it on a large portion of the Wikipedia corpus. The learned embeddings are then used to estimate which links on a page are most likely to move closer to a target article.

---

## Motivation

I've always thought Wikipedia Speedruns were an interesting problem because they combine graph traversal with human intuition. People usually don't know the shortest path between two pages—they make educated guesses based on what concepts seem related.

My goal with this project was to see if a machine learning model could learn a similar intuition by training directly on Wikipedia text.

Along the way, it also became a great opportunity to learn more about NLP, embedding models, PyTorch, and working with large datasets.

---

## How it Works

The project follows three main steps.

### Train Word Embeddings

A Skip-Gram with Negative Sampling (SGNS) model is trained on Wikipedia text to learn vector representations of words and concepts.

Concepts that appear in similar contexts should end up close together in the embedding space.

### Compare Candidate Links

When navigating Wikipedia, every outgoing link on the current page is converted into an embedding.

Each candidate is then compared to the target page using cosine similarity.

### Choose the Next Page

The algorithm selects the link that appears most semantically similar to the destination and repeats the process until the target page is reached.

This isn't guaranteed to find the shortest path, but it's an interesting way to approach navigation without explicitly searching the entire graph.

---

## Technologies Used

- Python
- PyTorch
- CUDA
- NumPy
- Wikipedia Corpus

---

## Features

- Skip-Gram with Negative Sampling implementation
- Custom vocabulary generation
- GPU training with CUDA
- Checkpoint saving/loading
- Cosine similarity search
- Wikipedia navigation algorithm

---

## Running the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train.py
```

Run the navigation algorithm

```bash
python main.py
```

---

## What I Learned

This project taught me much more than I expected going into it.

Beyond implementing the SGNS model itself, I spent a lot of time learning how to efficiently process large datasets, optimize PyTorch training pipelines, manage GPU memory, and structure a machine learning project from scratch.

It also gave me a much better understanding of how embedding models work and why they have become such a fundamental part of modern NLP.

---

## Future Work

Some ideas I'd like to explore next:

- Use phrase embeddings for full article titles instead of individual words
- Combine semantic similarity with graph search algorithms
- Compare performance against traditional shortest-path methods
- Experiment with transformer-based embeddings
- Build a visualization tool to display the path the algorithm takes
- Evaluate the algorithm on a larger collection of start/end page pairs

---

## Author

**Carson Apple**

Northwestern University  
Computer Engineering & Economics
