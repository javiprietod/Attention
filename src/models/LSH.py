# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
from src.models import PositionalEncoding

# other libraries
import math


class LSHAttention(torch.nn.Module):
    """
    LSH attention module.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        n_buckets: int = 32,
        partition_size: int = 40,
    ) -> None:
        """
        Constructor of the class SelfAttention.

        Args:
            embedding_dim: input channels of the module.
            num_heads: number of heads in the multi-head attention.
            n_buckets: number of buckets for the LSH module.
            If not a power of 2, it will be rounded to the previous power of 2.
            partition_size: size of the partitions.
        """
        seed = 42
        torch.manual_seed(seed)
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.n_buckets = n_buckets
        self.n_hyperplanes = int(math.log2(n_buckets))
        self.partition_size = partition_size

        self.q = torch.nn.Linear(embedding_dim, embedding_dim)
        self.v = torch.nn.Linear(embedding_dim, embedding_dim)

        self.hyperplanes = torch.randn(
            embedding_dim // num_heads + 1, self.n_hyperplanes
        )

    def get_buckets(self, x: torch.Tensor):
        """Function that maps the embeddings of words into buckets.

        Args:
            x: input tensor
                Dimensions: [batch, sequence, num_heads, Embedding_dim].

        Returns:
            buckets: buckets of the embeddings
                Dimensions: [batch, sequence, num_heads].
        """
        # Add a column of ones to the embeddings to allow for the bias term
        x = torch.cat(
            [
                x,
                torch.ones(
                    x.size(0), x.size(1), x.size(2), 1, device=x.device, dtype=x.dtype
                ),
            ],
            dim=3,
        )

        # Calculate the dot product between the embeddings and the hyperplanes, then threshold to get the buckets
        # Size: [B, N, H, n_hyperplanes]
        buckets = torch.einsum("bnhe,ey->bnhy", x, self.hyperplanes)
        buckets = (buckets >= 0).to(torch.float32)

        # Convert the binary representation of the buckets to decimal
        # Size: [B, N, H]
        buckets = torch.einsum(
            "bnhd,d->bnh",
            buckets,
            2 ** torch.arange(self.n_hyperplanes, device=x.device, dtype=x.dtype),
        )
        return buckets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method returns the output of the LSH module.

        Args:
            x: input tensor.
                Dimensions: [batch, sequence, channels].

        Returns:
            output of the self attention module.
                Dimensions: [batch, sequence, channels].
        """
        x_size = x.size()
        q: torch.Tensor = self.q(x)
        v: torch.Tensor = self.v(x)

        # Size: [B, N, H, E/H]
        q = q.view(
            x_size[0], x_size[1], self.num_heads, self.embedding_dim // self.num_heads
        )
        v = v.view(
            x_size[0], x_size[1], self.num_heads, self.embedding_dim // self.num_heads
        )

        # Get a matrix of the buckets of q [B, N, H]
        buckets = self.get_buckets(q)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # One hot encode the vectors to the bucket it's on [C, B, N, H]
        one_hot_buckets = F.one_hot(buckets.long(), num_classes=self.n_buckets).int()
        one_hot_buckets = torch.einsum("bnhc->cbhn", one_hot_buckets)

        # Multiply the vectors in the same bucket [C, B, H, E/H, E/H]
        q = q.masked_fill(one_hot_buckets.to(torch.bool).unsqueeze(4), 0)
        attention = torch.matmul(q, q.transpose(3, 4)) / math.sqrt(self.embedding_dim)

        # Remove the bucket dimension and apply softmax [B, H, N, N]
        attention = torch.sum(attention, dim=0)
        attention = F.softmax(attention, dim=-1)

        # Multiply the attention with the values and reshape [B, N, E]
        output = torch.matmul(attention, v)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(x_size[0], x_size[1], self.embedding_dim)
        )
        return output


class LSHModel(torch.nn.Module):
    """
    Model constructed used Block modules.
    """

    def __init__(
        self,
        sequence_length: int,
        vocab_to_int: dict[str, int],
        num_classes: int = 6,
        hidden_size: int = 32,
        encoders: int = 6,
        embedding_dim: int = 100,
        num_heads: int = 4,
        n_buckets: int = 64,
        **kwargs,
    ) -> None:
        """
        Constructor of the class CNNModel.

        Args:
            layers: output channel dimensions of the Blocks.
            sequence_length: input channels of the model.
            vocab_to_int: dictionary of vocabulary to integers.
            num_classes: output channels of the model.
            hidden_size: hidden size of the model.
            encoders: number of encoders in the model.
            embedding_dim: embedding dimension of the model.
            num_heads: number of heads in the multi-head attention.
            n_buckets: number of buckets for the LSH module.
        """

        super().__init__()
        self.vocab_to_int: dict[str, int] = vocab_to_int

        self.encoders: int = encoders

        # Embeddings
        self.embeddings = torch.nn.Embedding(
            len(vocab_to_int), embedding_dim, len(vocab_to_int) - 1
        )

        self.positional_encodings = PositionalEncoding(embedding_dim)

        # Normalization
        self.normalization = torch.nn.LayerNorm(embedding_dim)

        # self-attention
        self.self_attention = LSHAttention(embedding_dim, num_heads, n_buckets)

        # mlp
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, embedding_dim),
        )

        # classification
        self.model = torch.nn.Linear(embedding_dim * sequence_length, num_classes)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim * sequence_length),
            torch.nn.Linear(embedding_dim * sequence_length, hidden_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of logits.
        It is the output of the neural network.

        Args:
            inputs: batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            batch of logits. Dimensions: [batch, num_classes].
        """

        x = self.embeddings(inputs)
        x = self.positional_encodings(x)

        for _ in range(self.encoders):
            attention_x = self.self_attention(x)

            x = self.normalization(attention_x)

            x = self.fc(x) + x

            x = self.normalization(x)

        x = x.view(x.size(0), -1)

        return self.model(x)
