from word2vec_utils import (
    get_dataset,
    tokenize,
    vis_embeddings_wandb,
    vis_embeddings_matplotlib,
    reduce_dimensions,
)
from collections import Counter
from model import Model
import torch
import random
import numpy as np
import time
import wandb
import os
from tqdm import tqdm


class SkipGramDataset(torch.utils.data.Dataset):
    """Custom Dataset for training a SkipGram model."""

    def __init__(self, dataset, vocab, context_size):
        """
        Args:
        dataset: datasets.Dataset, the TinyStories dataset
        vocab: dict, a mapping from tokens to their indices
        context_size: int, the size of the context window
        """

        self.dataset = dataset
        self.vocab = vocab
        self.context_size = context_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """

        Here, we want to sample a random token from the text and a random context token
        from the context window around the token.

        Args:
        idx: int, the index of the sample in the dataset

        Returns:
        input_id: int, the index of the input token in the vocabulary
        context_id: int, the index of the context token in the vocabulary
        """

        # get text from the dataset
        sample = self.dataset[idx]

        # tokenize the text
        sample_tokens = tokenize(sample['text'])

        # edge case: if the sample is empty, return <UNK> tokens
        if len(sample_tokens) == 0:
            return self.vocab['<UNK>'], self.vocab['<UNK>']

        # 1. Randomly sample one target token to train the embedding on.
        # 2. For that target token, select a context token from its
        #    neighborhood window (defined by context_size).
        # 3. Convert the target and context tokens to IDs using self.vocab.
        #
        # Example input:
        # sample = "The is a fairly long sentence with many tokens."
        # Example output:
        # input_id = id_of('sentence')
        # context_id = id_of('long') or id_of('with') # context_size = 1
        #1
        target = random.randint(0, len(sample_tokens) - 1)
        
        #2
        window_start = max(0, target - self.context_size)
        window_end = min(len(sample_tokens) - 1, target + self.context_size)
        
        possible_indices = list(range(window_start, window_end + 1))
        possible_indices.remove(target)

        context = random.choice(possible_indices)
        trget_token = sample_tokens[target]
        ctext_token = sample_tokens[context]
        
        #3
        unk_id = self.vocab['<UNK>']
        input_id = self.vocab.get(trget_token, unk_id)
        context_id = self.vocab.get(ctext_token, unk_id)

        return input_id, context_id


def build_vocab(train_dataset, vocab_size=5000):
    """
    Build a vocabulary from the training dataset.

    Args:
    train_dataset: datasets.Dataset, the TinyStories training dataset
    vocab_size: int, the size of the vocabulary

    Returns:
    vocab: dict, a mapping from tokens to unique word_ids
    The vocabulary should contain the top `vocab_size` tokens in the training dataset, + the '<UNK>' token.
    """

    # 1. Convert all texts in train_dataset to tokens and track the count
    #    of each token.
    # 2. Convert the most common vocab_size tokens to IDs. All other, less
    #    common tokens are mapped to '<UNK>'.
    # 3. Return the vocabulary as a dictionary mapping tokens to IDs.
    vocab = {}
    count = Counter()
    
    #1
    for n in train_dataset:
        token = tokenize(n['text'])
        count.update(token)
        
    #2
    most_common_tok = count.most_common(vocab_size)
    
    #3
    vocab['<UNK>'] = 0
    for current_id, (token, count) in enumerate(most_common_tok, start=1):
        vocab[token] = current_id

    return vocab


def compute_loss(model, inputs, targets, loss_fn):
    # Obtain the model's predictions for the inputs and compute the loss
    # against the target context tokens.
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    return loss


def validation_step(model, validation_loader, loss_fn):
    model.eval()
    val_losses = []
    for inputs, targets in validation_loader:
        loss = compute_loss(model, inputs, targets, loss_fn)
        val_losses.append(loss.item())
    avg_val_loss = np.mean(val_losses)
    return avg_val_loss


def train(model, train_dataset, validation_dataset, vocab, wandb_online=True):
    # Hyperparameters and loss function.
    # A batch_size <= 128 is recommended to keep memory usage modest.
    batch_size = 128
    num_epochs = 7
    learning_rate = 0.001
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    wandb.init(
        project='word2vec-skipgram',
        config={
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'vocab_size': len(vocab),
            'embedding_dim': model.embedding_dim,
        },
        mode='online' if wandb_online else 'disabled',
    )

    current_step = 0
    initial_val_loss = validation_step(model, validation_loader, loss_fn)
    print(f'Initial validation loss: {initial_val_loss}')
    wandb.log({'val_loss': initial_val_loss})

    for epoch in tqdm(list(range(num_epochs))):
        model.train()

        start_time = time.time()

        # Training loop
        train_losses = []
        for inputs, targets in train_loader:
            # 1. Clear the gradients stored in the optimizer.
            # 2. Compute the loss via compute_loss.
            # 3. Backpropagate the loss.
            # 4. Update the model parameters.
            optimizer.zero_grad()
            loss = compute_loss(model, inputs, targets, loss_fn)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if current_step % 100 == 0:
                avg_train_loss = np.mean(train_losses[-10:])
                print(f'Train loss: {avg_train_loss}')
                wandb.log({'train_loss': avg_train_loss})
            current_step += 1

        # Validation loop
        avg_val_loss = validation_step(model, validation_loader, loss_fn)
        print((
            f'Epoch {epoch + 1}, Loss: {np.mean(train_losses)}, Val Loss: {avg_val_loss}, '
            f'Time Elapsed: {round(time.time() - start_time,2)}'
        ))
        wandb.log({'val_loss': avg_val_loss})

        # reduce the dimensions of the embeddings
        # save plots of the embeddings to disk (plots folder)
        reduced_embeddings = reduce_dimensions(model.get_embeddings(), d=2)
        vis_embeddings_matplotlib(epoch, vocab, reduced_embeddings)
    return


if __name__ == '__main__':
    use_wandb = os.environ.get('USE_WANDB', '0').strip() == '1'
    print(f'use_wandb: {use_wandb}')
    # 0) Set the random seed for reproducibility
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # 1) Load the raw datasets
    train_dataset = get_dataset('train')
    validation_dataset = get_dataset('validation')

    # 2) Build the vocabulary
    vocab = build_vocab(train_dataset)
    print(f'Vocabulary size: {len(vocab)}')

    assert '<UNK>' in vocab, "The '<UNK>' token is missing from the vocabulary"
    assert len(vocab) == 5001, "The vocabulary size is incorrect."

    # 3) Create the SkipGramDatasets
    context_size = 1  # Consider the 1 token to the left and right of the input token
    train_dataset = SkipGramDataset(train_dataset, vocab, context_size=1)
    validation_dataset = SkipGramDataset(validation_dataset, vocab, context_size=1)

    # 4) Initialize the model
    vocab_size = len(vocab)
    model = Model(vocab_size)

    # 5) Train the model
    # To log the plots to WandB, run `USE_WANDB=1 python train.py`
    train(model, train_dataset, validation_dataset, vocab, wandb_online=use_wandb)
