# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
import math
import numpy as np

from src.models import PositionalEncoding

import torch
import math

from src.models import PositionalEncoding
from torch.nn.functional import normalize

EMBEDDING_DIM: int = 256
NUM_CLASSES: int = 2


device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

class KernelizedLinformerAttention(torch.nn.Module):
    """
    Normalized Kernelized Attention with RPE using FFT.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mapping_dim: int | None = None,
        seq_len=68,
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
        self.seq_len = seq_len
        self.l = 48 # linformer k dimension of proyection of n
        self.sigma = 1 / (2**self.seq_len)

        # Matrices de proyección no entrenables E y F
        R = torch.randn((self.seq_len, self.l)) / math.sqrt(self.l)
        R = normalize(R, p=2, dim=1)  # Normalizar R

        # Convertir sigma a un tensor para poder usar torch.exp
        sigma_tensor = torch.full((1,), self.sigma, dtype=R.dtype, device=R.device)

        # E y F como atributos constantes
        self.register_buffer("E", sigma_tensor * R)
        self.register_buffer("F", torch.exp(-sigma_tensor) * R)

        self.q = torch.nn.Linear(embedding_dim, embedding_dim)
        self.k = torch.nn.Linear(embedding_dim, embedding_dim)
        self.v = torch.nn.Linear(embedding_dim, embedding_dim)

        self.mapping_dim = mapping_dim // num_heads if mapping_dim is not None else embedding_dim // num_heads
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

        q = q / torch.norm(q) # (batch, n, head, embedding_dim)
        k = k / torch.norm(k) # (batch, n, head, embedding_dim)
# 
        # c = torch.exp(self.pos_encoding)

        # Size of both: [B, S, N, M]
        phi_q = self.phi(q) # (batch, n, head, mapping_dim//heads)
        phi_k = self.phi(k) # (batch, n, head, mapping_dim//heads)

        # Size: [B, N, S, M] S = Head
        phi_q = phi_q.transpose(1, 2) # (batch, head, n, m)
        phi_k = phi_k.transpose(1, 2) # (batch, head, n, m)

        # Size: [B, N, S, M]
        A2 = phi_k

        # LARA
        # Proyección de K y V
        phi_k = torch.einsum("bhne,nk->bhke", phi_k, self.E) # (batch, head, k, m)
        v = torch.einsum("bnhe,nk->bkhe", v, self.F) # (batch, n, head, m)

        # Size: [B, N, S, E//N]
        v = v.transpose(1, 2) # (batch, head, n, m)

        # (b, head, map_dim, k) @ (batch, head, k, emb_dim) -> (batch, head, map_dim, emb_dim)
        A1 = torch.matmul(phi_k.transpose(2,3), v)
        
        num = torch.matmul(phi_q, A1) # (batch, head, n, emb_dim)
        
        den = torch.einsum("abcd,abcd->abc", phi_q, A2).unsqueeze(-1)

        output: torch.Tensor = num / den

        # Output: [B, S, E]
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

        # w: [N, M, E//N]
        # x: [B, S, N, E//N]
        output: torch.Tensor = torch.exp(
            torch.einsum("abc,deac->deab", self.weights, x)
        )

        # output: [B, S, N, M]
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


class KernelizedLinformerModel(torch.nn.Module):
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
        **kwargs
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
        self.self_attention = KernelizedLinformerAttention(
            embedding_dim, num_heads, mapping_dim, sequence_length
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

            x = self.normalization(attention_x)

            x = self.fc(x) + x

            x = self.normalization(x)

        x = x.view(x.size(0), -1)

        return self.model(x)




def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LinformerSelfAttention(torch.nn.Module):
    def __init__(self, dim, seq_len=68, k=256, heads=8, dim_head=None, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "Dimension must be divisible by the number of heads."
        self.eps = 1e-6
        self.seq_len = seq_len
        self.k = k
        self.heads = heads
        self.dim_head = (dim // heads) if dim_head is None else dim_head
        self.sigma = 1 / (2**self.seq_len)

        # Matrices de proyección no entrenables E y F
        R = torch.randn((self.seq_len, self.k)) / math.sqrt(self.k)
        R = normalize(R, p=2, dim=1)  # Normalizar R

        # Convertir sigma a un tensor para poder usar torch.exp
        sigma_tensor = torch.full((1,), self.sigma, dtype=R.dtype, device=R.device)

        # E y F como atributos constantes
        self.register_buffer("E", sigma_tensor * R)
        self.register_buffer("F", torch.exp(-sigma_tensor) * R)

        # Inicializar las capas lineales para Q, K, V
        self.to_q = torch.nn.Linear(dim, self.dim_head * heads, bias=False)
        self.to_k = torch.nn.Linear(dim, self.dim_head * heads, bias=False)
        self.to_v = torch.nn.Linear(dim, self.dim_head * heads, bias=False)

        # Capa de salida
        self.to_out = torch.nn.Linear(self.dim_head * heads, dim)

        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # b, n, _, _ = *x.shape, self.dim_head, self.heads
        (
            b,
            n,
            d,
        ) = x.shape

        # Pasar a través de las capas lineales Q, K, V
        q: torch.Tensor = self.to_q(x)
        k: torch.Tensor = self.to_k(x)
        v: torch.Tensor = self.to_v(x)

        # Redimensionar Q, K, V
        q = q.reshape(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = k.reshape(b, n, self.heads, self.dim_head).transpose(1, 2)
        v = v.reshape(b, n, self.heads, self.dim_head).transpose(1, 2)

        # Proyección de K y V
        k = torch.einsum("bhne,nk->bhke", k, self.E)
        v = torch.einsum("bhne,nk->bhke", v, self.F)

        # Cálculo de atención
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * (self.dim_head**-0.5)
        attn_prob = torch.softmax(attn_score, dim=-1)
        attn_out = torch.matmul(attn_prob, v)

        # Fusionar cabezas
        attn_out = attn_out.transpose(1, 2).reshape(b, n, self.heads * self.dim_head)
        attn_out = self.to_out(attn_out)

        return attn_out


class LinformerModel(torch.nn.Module):
    """
    Model constructed used Block modules.
    """

    def __init__(
        self,
        sequence_length: int,
        vocab_to_int: dict[str, int],
        num_classes: int = 6,
        hidden_size: int = 1024,
        encoders: int = 6,
        e: int = 100, # embedding dim
        n: int = 4, # number of heads
        **kwargs,
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

        # k and eps
        self.eps = 0.1
        #k = int(9*math.log(e)/(self.eps**2 - self.eps**3))
        k = int(9*math.log(e) / 0.9)
        # Embeddings
        self.embeddings = torch.nn.Embedding(
            len(vocab_to_int), e, len(vocab_to_int) - 1
        )

        self.positional_encodings = PositionalEncoding(e)

        # Normalization
        self.normalization = torch.nn.LayerNorm(e)

        # Linformer self-attention
        self.self_attention = LinformerSelfAttention(dim=e, k=k, heads=n)

        # mlp
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(e, hidden_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, e),
        )

        # classification
        self.model = torch.nn.Linear(e * sequence_length, num_classes)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(e * sequence_length),
            torch.nn.Linear(e * sequence_length, hidden_size),
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