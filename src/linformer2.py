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

        self.seq_len = seq_len
        self.k = k
        self.heads = heads
        self.dim_head = (dim // heads) if dim_head is None else dim_head
        self.sigma = 1 / (2 ** self.seq_len)

        # Matrices de proyección no entrenables E y F
        R = torch.randn((self.seq_len, self.k)) / math.sqrt(self.k)
        R = normalize(R, p=2, dim=1)  # Normalizar R

        # Convertir sigma a un tensor para poder usar torch.exp
        sigma_tensor = torch.full((1,), self.sigma, dtype=R.dtype, device=R.device)

        # E y F como atributos constantes
        self.register_buffer('E', sigma_tensor * R)
        self.register_buffer('F', torch.exp(-sigma_tensor) * R)

        # Inicializar las capas lineales para Q, K, V
        self.to_q = torch.nn.Linear(dim, self.dim_head * heads, bias=False)
        self.to_k = torch.nn.Linear(dim, self.dim_head * heads, bias=False)
        self.to_v = torch.nn.Linear(dim, self.dim_head * heads, bias=False)

        # Capa de salida
        self.to_out = torch.nn.Linear(self.dim_head * heads, dim)

        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # b, n, _, _ = *x.shape, self.dim_head, self.heads
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k  # type: ignore

        # Pasar a través de las capas lineales Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Redimensionar Q, K, V
        q = q.reshape(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = k.reshape(b, n, self.heads, self.dim_head).transpose(1, 2)
        v = v.reshape(b, n, self.heads, self.dim_head).transpose(1, 2)

        # Proyección de K y V
        k = torch.einsum('bhnd,nk->bhkd', k, self.E)
        v = torch.einsum('bhnd,nk->bhkd', v, self.F)

        # Cálculo de atención
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * (self.dim_head ** -0.5)
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
        hidden_size: int,
        vocab_to_int: dict[str, int],
        input_channels: int = 3,
        output_channels: int = 6,
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
        self.self_attention = LinformerSelfAttention(dim=embedding_dim, heads=nhead)

        # mlp
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, embedding_dim),
        )

        # classification
        self.model = torch.nn.Linear(embedding_dim * input_channels, output_channels)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim * input_channels),
            torch.nn.Linear(embedding_dim * input_channels, hidden_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_channels),
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