# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SelfAttention(torch.nn.Module):
    """
    Self attention module.
    """

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        """
        Constructor of the class SelfAttention.

        Args:
            input_channels: input channels of the module.
            output_channels: output channels of the module.
            num_heads: number of heads in the multi-head attention.
            mask: mask tensor for padding values. Dimensions: [batch, sequence].
        """

        # TODO
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.q = torch.nn.Linear(embedding_dim, embedding_dim)
        self.k = torch.nn.Linear(embedding_dim, embedding_dim)
        self.v = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        This method returns the output of the self attention module.

        Args:
            x: input tensor.
                Dimensions: [batch, sequence, channels].

        Returns:
            output of the self attention module.
        """

        # TODO
        q: torch.Tensor = self.q(x)
        k: torch.Tensor = self.k(x)
        v: torch.Tensor = self.v(x)

        q = q.view(
            x.size(0), x.size(1), self.num_heads, self.embedding_dim // self.num_heads
        )
        k = k.view(
            x.size(0), x.size(1), self.num_heads, self.embedding_dim // self.num_heads
        )
        v = v.view(
            x.size(0), x.size(1), self.num_heads, self.embedding_dim // self.num_heads
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attention = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(
            self.embedding_dim // self.num_heads
        )

        if mask is not None:
            attention = attention.masked_fill(
                mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attention = F.softmax(attention, dim=-1)

        output = torch.matmul(attention, v)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(x.size(0), x.size(1), self.embedding_dim)
        )

        return output


class EncoderModel(torch.nn.Module):
    """
    Model constructed used Block modules.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        vocab_to_int: dict[str, int],
        input_channels: int = 3,
        output_channels: int = 10,
        encoders: int = 6,
        embedding_dim: int = 100,
        nhead: int = 4,
    ) -> None:
        """
        Constructor of the class CNNModel.

        Args:
            layers: output channel dimensions of the Blocks.
            input_channels: input channels of the model.
        """

        # TODO
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
        self.self_attention = SelfAttention(embedding_dim, nhead)

        # mlp
        self.fc = torch.nn.Linear(embedding_dim, embedding_dim)

        # dropout
        self.dropout = torch.nn.Dropout(0.2)

        # classification
        self.model = torch.nn.Linear(embedding_dim * input_channels, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of logits.
        It is the output of the neural network.

        Args:
            inputs: batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            batch of logits. Dimensions: [batch, output_channels].
        """

        # TODO
        x = self.embeddings(inputs)
        x += self.positional_encodings(x)

        for _ in range(self.encoders):
            attention_x = self.self_attention(x, inputs==len(self.vocab_to_int) - 1)

            x = self.dropout(attention_x) + x

            x = self.normalization(x)

            x = self.fc(x) + x

            x = self.normalization(x)

        x = x.view(x.size(0), -1)

        return self.model(x)
