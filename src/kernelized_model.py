# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
import math
import numpy as np

from src.models import PositionalEncoding

EMBEDDING_DIM: int = 256
NUM_CLASSES: int = 2


device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


class KernelizedAttention(torch.nn.Module):
    """
    Normalized Kernelized Attention with RPE using FFT.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mapping_dim: int | None = None,
    ) -> None:
        """
        Constructor of the class SelfAttention.

        Args:
            embedding_dim: input channels of the module.
            num_heads: number of heads in the multi-head attention.
            mapping_dim: mapping dimension of the kernelized attention.
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

        # self.pos_encoding = torch.randn(
        #     2 * seq_length - 1, requires_grad=True, dtype=torch.cdouble
        # ).to(device)

        # self.n = seq_length
        # w = np.exp(np.pi * 2j / seq_length)

        # # We compute the FFT matrix
        # self.F_2n = torch.tensor(
        #     [
        #         [w ** (i * j) for j in range(2 * seq_length)]
        #         for i in range(2 * seq_length)
        #     ]
        # )
        # self.F_2n_inv = torch.tensor(
        #     [
        #         [w ** (-i * j) for j in range(2 * seq_length)]
        #         for i in range(2 * seq_length)
        #     ]
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method returns the output of the self attention module.

        Args:
            x: input tensor.
                Dimensions: [batch, sequence, channels].

        Returns:
            output of the self attention module.
        """

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

        q = q / torch.norm(q)
        k = k / torch.norm(k)
        #
        # c = torch.exp(self.pos_encoding)

        # Size of both: [B, N, H, M]
        phi_q = self.phi(q)
        phi_k = self.phi(k)

        # Size: [B, H, N, M]
        phi_q = phi_q.transpose(1, 2)
        phi_k = phi_k.transpose(1, 2)

        # Size: [B, H, N, E//H]
        v = v.transpose(1, 2)

        # Size: [B, H, N, M]
        A2 = phi_k

        # [B, H, N, M] x [B, H, N, E//H] = [B, H, N, M, E//H]
        A1 = torch.einsum("...d,...e->...de", phi_k, v)

        # D1: [B, H, N, M, E//H]
        # D1 = self.FFTmatmul(c, A1)

        # D2: [B, H, N, M]
        # D2 = self.FFTmatmul(c, A2)

        num = torch.einsum("abcd,abcde->abce", phi_q, A1)
        den = torch.einsum("abcd,abcd->abc", phi_q, A2).unsqueeze(-1)

        output: torch.Tensor = num / den

        # Output: [B, N, E]
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
        output = output * torch.exp(-(norm_x**2) / 2) / self.mapping_dim

        return output

    def FFTmatmul(self, c: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        """
        FFT matrix multiplication based on algorithm described in:
        https://alinush.github.io/2020/03/19/multiplying-a-vector-by-a-toeplitz-matrix.html

        Args:
            ...

        Returns:
            ...
        """
        # We compute the representation vector
        a_2n = torch.cat(
            (
                torch.tensor([c[self.n]]),
                torch.flip(c[: self.n], dims=[0]),
                torch.tensor([c[self.n]]),
                torch.flip(c[self.n + 1 :], dims=[0]),
            )
        )

        # DFT(a_2n) - [2n]
        dft_a_2n = torch.einsum("ab,b->a", self.F_2n, a_2n)

        # Pad n 0s in A1 to increase 3rd dimension to 2n
        F_2n = self.F_2n[:, : self.n]

        dft_mat = torch.einsum("ab,xyb...->xya...", F_2n, mat.to(torch.complex128))

        # DFT(A1) * DFT(a_2n)
        dft_mat = torch.einsum("a,xya...->xya...", dft_a_2n, dft_mat)

        # Inverse DFT
        output: torch.Tensor = torch.einsum("ab,xyb...->xya...", self.F_2n_inv, dft_mat)

        return output[: self.n].to(torch.float64)


class AttentionModelK(torch.nn.Module):
    """
    This class is the model for the attention models.
    """

    def __init__(
        self,
        attention: torch.nn.Module,
        sequence_length: int,
        vocab_to_int: dict[str, int],
    ) -> None:
        super().__init__()
        self.attention = attention
        self.vocab_to_int = vocab_to_int
        self.base_model = torch.nn.Embedding(
            len(vocab_to_int), EMBEDDING_DIM, len(vocab_to_int) - 1
        )
        self.linear = torch.nn.Linear(EMBEDDING_DIM * sequence_length, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.base_model(inputs)

        x = self.attention(x)

        x = x.view(x.size(0), -1)
        return self.linear(x)


class KernelizedModel(torch.nn.Module):
    """
    Model constructed used Block modules.
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
    ) -> None:
        """
        Constructor of the class CNNModel.

        Args:
            layers: output channel dimensions of the Blocks.
            sequence_length: input channels of the model.
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
