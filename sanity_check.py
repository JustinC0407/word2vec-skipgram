# Sanity check build_vocab, SkipGramDataset, model

from model import Model
from train import SkipGramDataset, build_vocab

import torch

if __name__ == "__main__":

    print('Sanity checking build_vocab...')

    dataset = []
    # add some text samples to the dataset
    dataset.append(
        {'text': ' '.join([f'{x}!_' for x in range(5, 0, -1)])}
    )  # 5!_ 4!_ 3!_ 2!_ 1!_
    dataset.append(
        {'text': ' '.join([f'{x}.-' for x in range(1, 5)])}
    )  # 1.- 2.- 3.- 4.-
    dataset.append({'text': '0 7'})

    # We expect the vocabulary to contain the following tokens:
    # 1, 2, 3, 4, <UNK>

    # 0, 5, 7 should be replaced with <UNK> token

    # build the vocabulary
    vocab = build_vocab(dataset, vocab_size=4)
    assert len(vocab) == 5, "The vocabulary size is incorrect, should be 4 + 1 (<UNK>)"
    assert (
        '<UNK>' in vocab
    ), f"The '<UNK>' token is missing from the vocabulary: {vocab}"

    vocab_keys = sorted(vocab.keys())
    assert vocab_keys == [
        '1',
        '2',
        '3',
        '4',
        '<UNK>',
    ], f"Vocabulary keys are incorrect: {vocab_keys}"

    print('build_vocab passed checks!')

    print('\nSanity checking SkipGramDataset...')

    # create the SkipGramDataset
    context_size = 1
    skipgram_dataset = SkipGramDataset(dataset, vocab, context_size=context_size)

    inverted_vocab = {v: k for k, v in vocab.items()}

    # check the dataset size
    assert len(skipgram_dataset) == 3

    # Test without UNK token
    for _ in range(10):
        input_idx, context_idx = skipgram_dataset[1]
        context_token = inverted_vocab[context_idx]
        input_token = inverted_vocab[input_idx]
        assert (
            int(context_token) - int(input_token)
        ) ** 2 == 1, f"Context and target tokens are not 1 token apart: {context_token}, {input_token}"

    # Test with UNK token
    input_idx, context_idx = skipgram_dataset[2]
    context_token = inverted_vocab[context_idx]
    input_token = inverted_vocab[input_idx]
    assert (
        context_token == '<UNK>' and input_token == '<UNK>'
    ), f"Context and target tokens are not <UNK>: {context_token}, {input_token}"

    print('SkipGramDataset passed checks!')

    print('\nSanity checking Model...')

    # Initialize the model
    model = Model(vocab_size=5)
    embedding_weights = model.get_embeddings()

    assert len(embedding_weights.shape) == 2, "The embedding matrix has the wrong shape"
    assert (
        embedding_weights.shape[0] == 5
    ), "The first rank of the embedding matrix is incorrect, it should be equal to the vocab size"

    sample_inputs = torch.tensor([1, 4])
    logits = model(sample_inputs)
    assert logits.shape == (2, 5), "The output logits have the wrong shape"

    print('Model passed checks!')

    print('\nAll checks passed!')
    print('Warning: These are basic sanity checks, and not exhaustive!!!')
