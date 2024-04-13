import torch
import math

from src.models import PositionalEncoding
from torch.nn.functional import normalize


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
        print(self.k)
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
        k = torch.einsum("bhnd,nk->bhkd", k, self.E)
        v = torch.einsum("bhnd,nk->bhkd", v, self.F)

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
        embedding_dim: int = 100,
        num_heads: int = 4,
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
        #k = int(9*math.log(embedding_dim)/(self.eps**2 - self.eps**3))
        k = int(9*math.log(embedding_dim) / 0.9)
        # Embeddings
        self.embeddings = torch.nn.Embedding(
            len(vocab_to_int), embedding_dim, len(vocab_to_int) - 1
        )

        self.positional_encodings = PositionalEncoding(embedding_dim)

        # Normalization
        self.normalization = torch.nn.LayerNorm(embedding_dim)

        # Linformer self-attention
        self.self_attention = LinformerSelfAttention(dim=embedding_dim, k=k, heads=num_heads)

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
