# Skip-gram Word2Vec (PyTorch)

A compact PyTorch implementation of Word2Vec trained with the skip-gram objective. The
model learns dense word embeddings from the [TinyStories](https://arxiv.org/abs/2305.07759)
dataset by predicting context words from a given target word, and visualizes how those
embeddings organize themselves over the course of training.

## What it does

Given a target token, the model predicts a neighboring context token drawn from a small
window around it. Training on many (target, context) pairs pushes words that appear in
similar contexts toward similar embedding vectors. Instead of negative sampling, this
implementation uses a full softmax cross-entropy loss over the vocabulary, which is simple
and works well at this vocabulary size.

## Architecture

The model (see `model.png`) is intentionally minimal:

- `nn.Embedding(5001, 64)` — a learnable 64-dimensional embedding for each of the 5001
  vocabulary entries (the 5000 most frequent tokens plus an `<UNK>` token).
- `nn.Linear(64, 5001, bias=False)` — projects an embedding to un-normalized scores
  (logits) over the full vocabulary.

The embedding and linear weights are initialized from a uniform distribution in
`[-0.1, 0.1]`. The forward pass looks up the target token's embedding and projects it to
per-vocabulary logits.

## Training

`train.py` builds a vocabulary of the most frequent tokens, constructs a `SkipGramDataset`
that samples (target, context) pairs on the fly, and trains the model with the Adam
optimizer and `nn.CrossEntropyLoss` over the logits. Training and validation losses are
logged, optionally to [Weights & Biases](https://wandb.ai/).

At the end of every epoch, the embedding matrix is projected to 2D with
[t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) and
saved as a scatter plot to the `plots/` folder.

## Embedding visualization

The `plots/` folder contains the 2D t-SNE projections saved after each epoch
(`epoch_0.png` through `epoch_6.png`), showing how the embedding space evolves during
training: points start diffuse and gradually form small clusters of syntactically and
semantically related words. `model.png` illustrates the model architecture.

## Install

```bash
pip install -r requirements.txt
```

Or run the convenience script:

```bash
bash setup.sh
```

## Run

Train the model (t-SNE plots are written to `plots/` each epoch):

```bash
python train.py
```

To also log training and validation loss curves to Weights & Biases:

```bash
USE_WANDB=1 python train.py
```

## Sanity checks

`sanity_check.py` runs a few lightweight checks on the core components — `build_vocab`,
the `SkipGramDataset` sampling logic, and the model's output shapes:

```bash
python sanity_check.py
```

These are basic smoke tests, not an exhaustive test suite.
