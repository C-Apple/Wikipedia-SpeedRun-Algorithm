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

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

If you only want to run the browser frontend without a trained model, the frontend can fall back to deterministic string similarity, but the full project and checkpoint loading require the dependencies above.

### Launch the browser speedrun test bench

Start the local web server from the repository root:

```bash
python -m src.speedrun_frontend --host 127.0.0.1 --port 8765
```

Then open this URL in your browser:

```text
http://127.0.0.1:8765
```

The UI lets you:

- enter a start Wikipedia URL/title and an end Wikipedia URL/title;
- optionally enter a local trained model path;
- run one speedrun with **Run speedrun**;
- run a 1000-run random benchmark with **Run 1000 random speedruns**;
- watch the timer, current page, highest-ranked hyperlink, current path, and top ranked candidates update while a run is active;
- reload and graph saved results.

Model path formats supported by the frontend:

```text
runs/sgns_wiki_v2
runs/sgns_wiki_v2/checkpoint.pt
/absolute/path/to/a/checkpoint/folder
/absolute/path/to/checkpoint.pt
```

When you point at a `checkpoint.pt` file, keep the matching `vocab.json` beside it. When you point at a folder, that folder should contain both `checkpoint.pt` and `vocab.json`. Leave the model field blank to use the string-similarity fallback scorer.

Results are appended to:

```text
runs/speedrun_bench/results.jsonl
```

### Optional API checks

List saved results:

```bash
curl http://127.0.0.1:8765/api/results
```

Start one run through the API:

```bash
curl -X POST http://127.0.0.1:8765/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"start":"Python (programming language)","target":"Artificial intelligence","max_steps":25,"link_limit":500,"runs":1,"mode":"single"}'
```

Copy the returned `id` and poll job state:

```bash
curl http://127.0.0.1:8765/api/jobs/<job-id>
```

### Troubleshooting

- If port `8765` is already in use, choose another port, for example `python -m src.speedrun_frontend --port 9000`.
- The model path must be readable from the machine running the Python server. Browser file paths from another computer will not work.
- Real speedruns call the Wikipedia API, so the frontend needs internet access.
- A 1000-run benchmark can take a long time because each run fetches live Wikipedia links.

### Run automated tests

```bash
pytest
```

### Run tests from a browser

This repository includes a small standard-library web frontend for launching the pytest suite and viewing the captured output.

```bash
python -m src.test_frontend
```

Open `http://127.0.0.1:8765`, choose the test files you want to run, and click **Run selected tests**.

You can still run the full suite directly from the terminal:

```bash
pytest
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

#TODO
