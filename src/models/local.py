import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import PositionalEncoding


class LocalAttention(nn.Module):
    """
    Local window self attention module using loops.
    """

    def __init__(self, embedding_dim: int, num_heads: int, window_size: int) -> None:
        """
        Constructor of the class LocalAttention using loops.

        Args:
            embedding_dim: embedding dimension of the model.
            num_heads: number of heads in the multi-head attention.
            window_size: size of the local attention window.
        """

        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Linear layers for queries, keys and values
        self.q = nn.Linear(embedding_dim, embedding_dim)
        self.k = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method returns the output of the local window self attention module using loops.

        Args:
            x: input tensor.
                Dimensions: [batch, sequence, embedding_dim].

        Returns:
            output of the local self attention module.
        """

        batch_size, sequence_length, _ = x.size()

        q: torch.Tensor = self.q(x)
        k: torch.Tensor = self.k(x)
        v: torch.Tensor = self.v(x)

        # Transform Q, K and V to have shape [B, H, N, E/H]
        q = q.view(batch_size, self.num_heads, sequence_length, self.head_dim)
        k = k.view(batch_size, self.num_heads, sequence_length, self.head_dim)
        v = v.view(batch_size, self.num_heads, sequence_length, self.head_dim)

        # Initialize attention scores with zeros and dimensions [B, H, N, 1, N]
        attention_scores = torch.zeros(
            batch_size,
            self.num_heads,
            sequence_length,
            1,
            sequence_length,
            device=x.device,
        )

        # Transpose and reshape Q -> [B, H, N, 1, E/H] and K -> [B, H, E/H, N]
        k = k.transpose(-2, -1)
        q = q.unsqueeze(3)

        for i in range(sequence_length):
            start = max(0, i - self.window_size)
            end = min(sequence_length, i + self.window_size)

            # Calculate attention scores multiplying Q and K [B, H, 1, E/H] x [B, H, E/H, W] -> [B, H, 1, W]
            attention_scores[:, :, i, :, start:end] = torch.matmul(
                q[:, :, i, :, :], k[:, :, :, start:end]
            )

        # Reshape attention scores to [B, H, N, N]
        attention_scores = attention_scores.squeeze(3)

        # Normalize attention scores
        attention_scores = attention_scores * self.scale

        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply attention to the values [B, H, N, N] x [B, H, N, E/H] -> [B, H, N, E/H]
        output = torch.matmul(attention_probs, v)

        # Transpose and reshape output [B, H, N, E/H] -> [B, N, H, E/H] -> [B, N, E]
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.embedding_dim)
        )

        return output


class LocalAttentionUnFold(nn.Module):
    """
    Local window self attention module using unfold.
    """

    def __init__(
        self, embedding_dim: int, num_heads: int, window_size: int, sequence_length: int
    ) -> None:
        """
        Constructor of the class LocalAttention using unfold.

        Args:
            embedding_dim: embedding dimension of the model.
            num_heads: number of heads in the multi-head attention.
            window_size: size of the local attention window.
            sequence_length: length of the sequence.
        """

        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.padding = (window_size - 1) // 2

        # Linear layers for queries, keys and values
        self.q = nn.Linear(embedding_dim, embedding_dim)
        self.k = nn.Linear(embedding_dim, embedding_dim)
        self.v = nn.Linear(embedding_dim, embedding_dim)

        # Create indices for padding
        self.i = torch.tensor(
            [
                [i for _ in range(-self.padding, self.padding + 1)]
                for i in range(sequence_length)
            ]
        ).view(-1)
        self.j = torch.tensor(
            [
                [j + i for j in range(-self.padding, self.padding + 1)]
                for i in range(sequence_length)
            ]
        ).view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method returns the output of the self attention module using unfold.

        Args:
            x: input tensor.
                Dimensions: [batch, sequence, embedding_dim].

        Returns:
            output of the self attention module.
        """
        batch_size, sequence_length, _ = x.size()

        q: torch.Tensor = self.q(x)
        k: torch.Tensor = self.k(x)
        v: torch.Tensor = self.v(x)

        # Transform Q, K and V to have shape [B, H, N, E/H]
        q = q.view(batch_size, self.num_heads, sequence_length, self.head_dim)
        k = k.view(batch_size, self.num_heads, sequence_length, self.head_dim)
        v = v.view(batch_size, self.num_heads, sequence_length, self.head_dim)

        # Unfold K to have shape [B, H, N, E/H, W]
        k_unf = F.pad(k, (0, 0, self.padding, self.padding), "constant", 0.0).unfold(
            dimension=2, size=self.window_size, step=1
        )

        # Multiply Q and K [B, H, N, E/H, W] x [B, H, N, E/H] -> [B, H, N, W]
        att = torch.einsum("...cde,...cd->...ce", k_unf, q) * self.scale

        # Apply softmax and reshape [B, H, N, W] -> [B, H, NxW]
        attn = F.softmax(att, dim=-1).view(att.shape[0], att.shape[1], -1)

        # Create tensor to store the attention values [B, H, N, N + 2*padding]
        qk = torch.zeros(
            att.shape[0],
            att.shape[1],
            att.shape[2],
            att.shape[2] + self.padding * 2,
            device=x.device,
        )

        # Fill the tensor with the attention values [B, H, N, N + 2*padding]
        qk[:, :, self.i, self.j] = attn.view(attn.shape[0], attn.shape[1], -1)

        # Remove padding [B, H, N, N + 2*padding] -> [B, H, N, N]
        qk = qk[:, :, :, self.padding : -self.padding]

        # Apply attention to the values [B, H, N, N] x [B, H, N, E/H] -> [B, H, N, E/H]
        output = torch.matmul(qk, v)

        # Transpose and reshape output [B, H, N, E/H] -> [B, N, H, E/H] -> [B, N, E]
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.embedding_dim)
        )

        return output


class LocalModel(nn.Module):
    """
    Model constructed using Local window self attention.
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
        **kwargs
    ) -> None:
        """
        Constructor of the class LocalModel.

        Args:
            sequence_length: length of the sequence.
            vocab_to_int: dictionary of vocabulary to integers.
            hidden_size: hidden size of the model.
            num_classes: output channels of the model.
            embedding_dim: embedding dimension of the model.
            encoders: number of encoders in the model.
            num_heads: number of heads in the multi-head attention.
            window_size: size of the local attention window.
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

        # Local self-attention using loops
        # self.self_attention = LocalAttention(
        #     embedding_dim, num_heads, window_size
        # )

        # Local self-attention using unfold (faster)
        self.self_attention = LocalAttentionUnFold(
            embedding_dim, num_heads, window_size, sequence_length
        )

        # MLP
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, embedding_dim),
        )

        # Classification
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
        This method returns a batch of predictions.
        Args:
            inputs: batch of texts.
                Dimensions: [batch, sequence]

        Returns:
            batch of predictions.
                Dimensions: [batch, num_classes].
        """

        x = self.embeddings(inputs)
        x = self.positional_encodings(x)

        for _ in range(self.encoders):
            attention_x = self.self_attention(x)

            x = self.normalization(attention_x) + x

            fc_x = self.fc(x)

            x = self.normalization(fc_x) + x

        x = x.view(x.size(0), -1)

        return self.model(x)
