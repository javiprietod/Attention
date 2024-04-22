import torch
import math

from src.models import PositionalEncoding
from torch.nn.functional import normalize


def init_(tensor):
    """
    Initialize the given tensor with uniform distribution scaled by the inverse
    square root of its last dimension.
    """
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LinformerSelfAttention(torch.nn.Module):
    """
    Linformer self-attention module.
    """

    def __init__(
        self,
        embedding_dim=128,
        sequence_length=68,
        num_heads=8,
        dim_head=None,
        dropout=0.0,
    ):
        """
        Constructor of the class LinformerSelfAttention.

        Args:
            embedding_dim: dimension of the input embeddings.
            sequence_length: input channels of the module.
            num_heads: number of heads in the multi-head attention.
            dim_head: dimension of the head.
            dropout: dropout rate.
        """

        super().__init__()
        assert (
            embedding_dim % num_heads == 0
        ), "Dimension must be divisible by the number of heads."

        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.dim_head = (embedding_dim // num_heads) if dim_head is None else dim_head

        # l and eps = l is the linformer's dimension of proyection of n (sequence length)
        self.eps = 0.9
        self.l = int(5 * math.log(sequence_length * embedding_dim) / self.eps)

        # Proyection matrices E and F
        self.sigma = 1 / (2**self.sequence_length)
        R = torch.randn((self.sequence_length, self.l)) / math.sqrt(self.l)
        R = normalize(R, p=2, dim=1)  # Normalize R

        # Convert sigma to tensor
        sigma_tensor = torch.full((1,), self.sigma, dtype=R.dtype, device=R.device)

        # E and F as non-learnable parameters
        self.register_buffer("E", sigma_tensor * R)
        self.register_buffer("F", torch.exp(-sigma_tensor) * R)

        # Initizalize linear layers Q, K, V
        self.to_q = torch.nn.Linear(
            embedding_dim, self.dim_head * num_heads, bias=False
        )
        self.to_k = torch.nn.Linear(
            embedding_dim, self.dim_head * num_heads, bias=False
        )
        self.to_v = torch.nn.Linear(
            embedding_dim, self.dim_head * num_heads, bias=False
        )

        # Output linear layer and dropout
        self.to_out = torch.nn.Linear(self.dim_head * num_heads, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Processes the input tensor through multiple layers of the self-attention mechanism,
        including linear transformation of inputs into queries (Q), keys (K), and values (V),
        followed by reshaping and transposing for attention computation. Projection is applied
        to keys and values before calculating attention scores and probabilities. The output
        is then produced by combining the heads and applying a final linear transformation.

        Args:
            x (torch.Tensor): Input tensor to the self-attention module.
                Dimensions should be [batch, sequence, channels], where:
                - batch is the batch size,
                - sequence is the sequence length,
                - channels is the number of channels.

        Returns:
            torch.Tensor: The output tensor from the self-attention module.
                The output retains the dimensions [batch, sequence, channels], but
                the channels may be transformed depending on the model configuration.
        """

        b, n, _ = x.shape  # (B, N, E)

        # Apply linear layers Q, K, V -> (B, N, E)
        q: torch.Tensor = self.to_q(x)
        k: torch.Tensor = self.to_k(x)
        v: torch.Tensor = self.to_v(x)

        # Reshape Q, K, V -> (B, H, N, E//H)
        q = q.reshape(b, n, self.num_heads, self.dim_head).transpose(1, 2)
        k = k.reshape(b, n, self.num_heads, self.dim_head).transpose(1, 2)
        v = v.reshape(b, n, self.num_heads, self.dim_head).transpose(1, 2)

        # Proyection of K y V -> (B, H, L, E//H)
        k = torch.einsum("bhne,nk->bhke", k, self.E)
        v = torch.einsum("bhne,nk->bhke", v, self.F)

        # Calculate attention
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * (
            self.dim_head**-0.5
        )  # (B, H, N, L)
        attn_prob = torch.softmax(attn_score, dim=-1)  # (B, H, N, L)
        attn_out = torch.matmul(attn_prob, v)  # (B, H, N, E//H)

        # Mix heads
        attn_out = attn_out.transpose(1, 2).reshape(
            b, n, self.num_heads * self.dim_head
        )  # (B, N, E)
        attn_out = self.to_out(attn_out)  # (B, N, E)

        return attn_out


class LinformerModel(torch.nn.Module):
    """
    A Linformer-based model integrating self-attention mechanisms with additional
    MLP layers for classification. This model is designed to process sequences with
    vocabulary-encoded inputs.
    """

    def __init__(
        self,
        sequence_length: int,
        vocab_to_int: dict[str, int],
        num_classes: int = 6,
        hidden_size: int = 1024,
        encoders: int = 6,
        embedding_dim: int = 100,
        num_heads: int = 4,
        **kwargs,
    ) -> None:
        """
        Initializes the LinformerModel.

        Args:
            sequence_length (int): Length of input sequences.
            vocab_to_int (dict[str, int]): Mapping from vocabulary to integers.
            num_classes (int, optional): Number of output classes.
            hidden_size (int, optional): Size of the hidden layers in MLP.
            encoders (int, optional): Number of encoder layers. Default is 6.
            embedding_dim (int, optional): Dimensionality of the embedding layer.
            num_heads (int, optional): Number of attention heads in Linformer.
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

        # Linformer self-attention
        self.self_attention = LinformerSelfAttention(
            embedding_dim=embedding_dim, num_heads=num_heads
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
            batch of logits. Dimensions: [batch, num_classes].
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
