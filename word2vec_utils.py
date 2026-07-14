""" Utility functions for training a SkipGram model using the TinyStories dataset. """

import os
import string
import random
import torch

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import wandb

# We will be using Huggingface's datasets library to load the TinyStories dataset.
# https://huggingface.co/docs/datasets/en/quickstart
from datasets import load_dataset


def get_dataset(shard, max_samples=100000):
    """
    Load the TinyStories dataset from Huggingface's datasets library.

    Args:
    shard: str, either 'train' or 'validation'
    max_samples: int, the maximum number of samples to load from the shard

    Returns:
    dataset: datasets.Dataset
    """

    assert shard in ['train', 'validation']

    # Load dataset from TinyStories (https://arxiv.org/abs/2305.07759)
    dataset = load_dataset("roneneldan/TinyStories")

    # return the first max_samples samples from the shard
    shard_data = dataset[shard]

    to_select = min(max_samples, len(shard_data))

    return dataset[shard].select(range(to_select))


def tokenize(text):
    """Normalize and tokenizer a string of text into a list of words.
    Args:
    text: str, the input text
    Returns:
    tokens: list of str, the list of normalized tokens
    """

    # remove punctuation and non-ascii characters
    text = ''.join([c if c not in string.punctuation else ' ' for c in text])
    text = ''.join([c if ord(c) < 128 else ' ' for c in text])

    # convert to lowercase and split into words
    tokens = text.lower().split()
    return tokens


def reduce_dimensions(embeddings, d=2):
    """
    Reduce the dimensions of the embeddings using t-SNE.

    Args:
    embeddings: torch.Tensor, the embeddings matrix
    d: int, the number of dimensions to reduce to

    Returns:
    reduced_embeddings: numpy.ndarray, the reduced embeddings matrix
    """

    embeddings = embeddings.detach().cpu().numpy()
    tsne = TSNE(n_components=d, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings


def vis_embeddings_matplotlib(epoch, vocab, embeddings_2d, folder="plots"):
    """
    Visualize the embeddings using Matplotlib.

    Args:
    epoch: int, the current epoch
    vocab: dict, a mapping from tokens to their indices
    embeddings: torch.Tensor, the reduced embeddings matrix
    """

    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(20, 20))

    # plot all embeddings
    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]

    plt.scatter(x, y)

    for token, token_id in vocab.items():
        if token_id % 10 == 0:
            x1, y1 = embeddings_2d[token_id]
            plt.annotate(token, (x1, y1))

    plt.title(f"Embeddings at epoch {epoch}")
    plt.savefig(f"{folder}/epoch_{epoch}.png")


def vis_embeddings_wandb(epoch, vocab, embeddings_2d):
    """
    Visualize the embeddings using WandB

    Args:
    epoch: int, the current epoch
    vocab: dict, a mapping from tokens to their indices
    embeddings: torch.Tensor, the embeddings matrix
    """

    table_cols = ["text", "x1", "x2"]
    table = wandb.Table(columns=table_cols)
    for token, token_id in vocab.items():
        table.add_data(token, *embeddings_2d[token_id])

    wandb.log({f"scatter_epoch_{epoch}": wandb.plot.scatter(table, "x1", "x2", "text")})


if __name__ == "__main__":
    # Load the TinyStories dataset
    train_dataset = get_dataset('train')
    validation_dataset = get_dataset('validation')

    # Print the first sample from the training and validation datasets
    print("Training dataset:")
    print(train_dataset[:1])
    print("\nValidation dataset:")
    print(validation_dataset[:1])
