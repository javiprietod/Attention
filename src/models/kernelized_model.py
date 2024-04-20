# deep learning libraries
import torch

# other libraries
from src.models import PositionalEncoding

from typing import Optional
import math

EMBEDDING_DIM: int = 256
NUM_CLASSES: int = 2


device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


class KernelizedAttention(torch.nn.Module):
    """
    Normalized Kernelized Attention.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mapping_dim: Optional[int] = None,
    ) -> None:
        """
        Constructor of the class SelfAttention.

        Args:
            embedding_dim: embedding dimension of the model.
            num_heads: number of heads in the multi-head attention.
            mapping_dim: dimension of the phi-mapping.
        """

        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.q = torch.nn.Linear(embedding_dim, embedding_dim)
        self.k = torch.nn.Linear(embedding_dim, embedding_dim)
        self.v = torch.nn.Linear(embedding_dim, embedding_dim)

        self.mapping_dim = (
            mapping_dim // num_heads
            if mapping_dim is not None
            else embedding_dim // num_heads
        )
        self.weights = torch.randn(
            num_heads, self.mapping_dim, embedding_dim // num_heads
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method returns the output of the kernelized attention module.

        Args:
            x: input tensor.
                Dimensions: [batch, sequence, embedding_dim].

        Returns:
            output of the kernelized attention module.
        """

        q: torch.Tensor = self.q(x)
        k: torch.Tensor = self.k(x)
        v: torch.Tensor = self.v(x)

        # q, k, v: FROM [B, N, E] TO [B, N, H, E//H]
        q = q.view(
            x.size(0), x.size(1), self.num_heads, self.embedding_dim // self.num_heads
        )
        k = k.view(
            x.size(0), x.size(1), self.num_heads, self.embedding_dim // self.num_heads
        )
        v = v.view(
            x.size(0), x.size(1), self.num_heads, self.embedding_dim // self.num_heads
        )

        q = q / torch.norm(q)
        k = k / torch.norm(k)

        # phi_q, phi_k: FROM [B, N, H, E//N] TO [B, N, H, M]
        phi_q = self.phi(q)
        phi_k = self.phi(k)

        # phi_q, phi_k: FROM [B, N, H, M] TO [B, H, N, M]
        phi_q = phi_q.transpose(1, 2)
        phi_k = phi_k.transpose(1, 2)

        # v: FROM [B, N, H, E//N] TO [B, H, N, E//H]
        v = v.transpose(1, 2)

        # A2: [B, H, N, M]
        A2 = phi_k

        # A1: [B, H, N, M] x [B, H, N, E//H] = [B, H, N, M, E//H]
        A1 = torch.einsum("...d,...e->...de", phi_k, v)

        # num: [B, H, N, M] x [B, H, N, M, E//H] = [B, H, N, E//H]
        num = torch.einsum("abcd,abcde->abce", phi_q, A1)

        # den: [B, H, N, M] x [B, H, N, M] = [B, H, N]
        den = torch.einsum("abcd,abcd->abc", phi_q, A2).unsqueeze(-1)

        # output: [B, H, N, E//H]
        output: torch.Tensor = num / den

        # output: FROM [B, H, N, E//H] TO [B, N, E]
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(x.size(0), x.size(1), self.embedding_dim)
        )
        return output

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method applies the exponential function to the inputs
        as non-linear function of the kernelized .

        Args:
            x: input tensor.
                Dimensions: [batch, sequence, num_heads, embedding_dim // num_heads]

        Returns:
            output of the kernelized attention module.
        """
        norm_x = torch.norm(x)

        # w: [H, M, E//H]
        # x: [B, N, H, E//H]
        output: torch.Tensor = torch.exp(
            torch.einsum("abc,deac->deab", self.weights, x)
        )

        # output: [B, N, H, M]
        output = output * torch.exp(-(norm_x**2) / 2) / math.sqrt(self.mapping_dim)

        return output


class KernelizedModel(torch.nn.Module):
    """
    Model constructed based on KernelizedAttention.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_to_int: dict[str, int],
        sequence_length: int = 3,
        num_classes: int = 6,
        encoders: int = 6,
        embedding_dim: int = 100,
        num_heads: int = 4,
        mapping_dim: int = 0,
        **kwargs,
    ) -> None:
        """
        Constructor of the class KernelizedModel.

        Args:
            hidden_size: hidden size of the model.
            vocab_to_int: dictionary of vocabulary to integers.
            sequence_length: input channels of the model.
            num_classes: output channels of the model.
            encoders: number of encoders in the model.
            embedding_dim: embedding dimension of the model.
            num_heads: number of heads in the multi-head attention.
            mapping_dim: dimension of the phi-mapping.
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
        self.self_attention = KernelizedAttention(embedding_dim, num_heads, mapping_dim)

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
