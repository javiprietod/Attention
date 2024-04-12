import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import PositionalEncoding


class LocalAttention(nn.Module):
    """
    Local self attention module.
    """

    def __init__(self, embedding_dim: int, num_heads: int, window_size: int) -> None:
        """
        Constructor of the class LocalSelfAttention.

        Args:
            embedding_dim: embedding dimension of the input.
            num_heads: number of heads in the multi-head attention.
            window_size: size of the local attention window.
        """

        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(embedding_dim, embedding_dim)
        self.k = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method returns the output of the local self attention module.

        Args:
            x: input tensor.
                Dimensions: [batch, sequence, channels].
            mask: mask tensor for padding values.
                Dimensions: [batch, sequence].

        Returns:
            output of the local self attention module.
        """

        batch_size, sequence_length, _ = x.size()

        q: torch.Tensor = self.q(x)
        k: torch.Tensor = self.k(x)
        v: torch.Tensor = self.v(x)

        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Initialize attention scores with zeros
        attention_scores = torch.zeros(
            batch_size,
            self.num_heads,
            sequence_length,
            1,
            sequence_length,
            device=x.device,
        )

        k = k.transpose(-2, -1)
        q = q.unsqueeze(3)

        for i in range(sequence_length):
            start = max(0, i - self.window_size)
            end = min(sequence_length, i + self.window_size)
            attention_scores[:, :, i, :, start:end] = torch.matmul(q[:,:,i,:,:], k[:, :, :, start:end])

        attention_scores = attention_scores.squeeze(3)

        # Normalize attention scores
        attention_scores = attention_scores * self.scale

        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply attention to the values
        output = torch.matmul(attention_probs, v)

        # Transpose and reshape output
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.embedding_dim)
        )

        return output


class LocalSelfAttentionUnFold(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(embedding_dim, embedding_dim)
        self.k = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.size()

        q: torch.Tensor = self.q(x)
        k: torch.Tensor = self.k(x)
        v: torch.Tensor = self.v(x)

        # Transform to Q, K, V
        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        # Transpose to get dimensions batch_size, num_heads, sequence_length, head_dim
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use unfold to create sliding windows
        k_unfolded = k.unfold(dimension=2, size=self.window_size, step=1).contiguous()
        v_unfolded = v.unfold(dimension=2, size=self.window_size, step=1).contiguous()

        # Pad q, k, v unfolded tensors to have the same sequence length
        padding = (self.window_size - 1) // 2
        q_padded = F.pad(q, (0, 0, padding, padding), "constant", 0.0)

        q_unfolded = q_padded.unfold(
            dimension=2, size=self.window_size, step=1
        ).contiguous()

        # Calculate scores
        attention_scores = (
            torch.einsum("bnqdh,bnkdh->bnqk", q_unfolded, k_unfolded) * self.scale
        )

        # Apply softmax
        attn = F.softmax(attention_scores, dim=-1)

        # Apply attention to the values
        output = torch.einsum("bnqk,bnkdh->bnqd", attn, v_unfolded)

        # Combine heads
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.embedding_dim)
        )

        return output


class LocalModel(nn.Module):
    """
    Model constructed used Block modules.
    """

    def __init__(
        self,
        sequence_length: int,
        vocab_to_int: dict[str, int],
        hidden_size: int,
        num_classes: int = 6,
        embedding_dim: int = 100,
        encoders: int = 6,
        num_heads: int = 4,
        window_size: int = 5,
    ) -> None:
        """
        Constructor of the class CNNModel.

        Args:
            layers: output channel dimensions of the Blocks.
            input_channels: input channels of the model.
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
        self.self_attention = LocalAttention(
            embedding_dim, num_heads, window_size
        )

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
            batch of logits. Dimensions: [batch, output_channels].
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
