""" Skip-gram Word2Vec model. """

import torch


class Model(torch.nn.Module):
    """Skip-gram Word2Vec model."""

    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.embedding_dim = 64
        # The architecture (see model.png) is a single torch.nn.Embedding
        # (vocab size by embedding dim) followed by a single torch.nn.Linear
        # layer with no bias. The embedding matrix is initialized with a
        # uniform distribution between -0.1 and 0.1.
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim)
        self.linear = torch.nn.Linear(in_features=self.embedding_dim, out_features=vocab_size, bias=False)

        torch.nn.init.uniform_(self.embeddings.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(self.linear.weight, a=-0.1, b=0.1)
        return

    def forward(self, input_ids):
        """Forward pass of the model.

        Args:
        input_ids: torch.Tensor, the input token indices [batch_size]

        Returns:
        logits: torch.Tensor, the output logits [batch_size, vocab_size]
        """
        # Look up the input embeddings and project to vocabulary logits (see model.png).
        embeds = self.embeddings(input_ids)
        logits = self.linear(embeds)
        return logits

    def get_embeddings(self):
        """Return the embedding matrix.
        Returns:
        torch.Tensor, the embedding [vocab_size, embedding_dim]
        """
        return self.embeddings.weight
