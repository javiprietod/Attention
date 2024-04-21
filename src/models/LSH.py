# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
import math

class LSHmodule(torch.nn.Module):

    """
    LSH attention module.
    """

    def __init__(self, embedding_dim: int, num_heads: int, n_buckets:int = 32, partition_size:int = 40) -> None:
        """
        Constructor of the class SelfAttention.

        Args:
            embedding_dim: input channels of the module.
            num_heads: number of heads in the multi-head attention.
            n_buckets: number of buckets for the LSH module. 
            If not a power of 2, it will be rounded to the previous power of 2.
        """

        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.n_buckets = n_buckets
        self.n_hyperplanes = int(math.log2(n_buckets))
        self.partition_size = partition_size

        self.q = torch.nn.Linear(embedding_dim, embedding_dim)
        self.v = torch.nn.Linear(embedding_dim, embedding_dim)
        
        self.hyperplanes = torch.randn(embedding_dim//num_heads + 1, self.n_hyperplanes)

    def get_buckets(self, x: torch.Tensor):
        """Function that maps the embeddings of words into buckets.

        Args:
            x: input tensor
                Dimensions: [batch, sequence, num_heads, Embedding_dim].

        Returns:
            buckets: buckets of the embeddings
                Dimensions: [batch, sequence, num_heads].
        """
        # Add a column of ones to the embeddings
        x = torch.cat([x, torch.ones(x.size(0), x.size(1),x.size(2), 1)], dim=3)

        # Calculate the dot product between the embeddings and the hyperplanes to get the buckets
        hyperplanes = self.hyperplanes

        # Size: [batch, sequence, num_heads, n_hyperplanes]
        buckets = torch.einsum('bnhe,ey->bnhy', x, hyperplanes)
        buckets = torch.where(buckets >= 0, torch.ones_like(buckets), torch.zeros_like(buckets))
        
        # Convert the binary representation of the buckets to decimal
        buckets = torch.einsum('bshn,n->bsh', buckets, 2 ** torch.arange(self.n_hyperplanes).to(torch.float32))

        return buckets
    
    def adjust_buckets(self, buckets):
        partition_size = self.partition_size
        # Flatten the buckets tensor
        flattened_buckets = buckets.view(-1)
        
        # Count the frequency of each element
        counts = flattened_buckets.bincount()
        new_buckets = torch.zeros_like(flattened_buckets)
        new_value = 0
        for i, count in enumerate(counts):
            if count <= partition_size:
                new_buckets[flattened_buckets == i] = new_value
                new_value += 1
            else:
                while count > 0:
                    new_buckets[flattened_buckets == i] = new_value
                    flattened_buckets[flattened_buckets == i] = new_value
                    new_value += 1
                    count -= partition_size


        # Reshape new_buckets to match the shape of buckets
        new_buckets = new_buckets.view(buckets.size())
        return new_buckets

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

        # Size: [batch, sequence, num_heads, embedding_dim // num_heads]
        q = q.view(
            x_size[0], x_size[1], self.num_heads, self.embedding_dim // self.num_heads
        )
        v = v.view(
            x_size[0], x_size[1], self.num_heads, self.embedding_dim // self.num_heads
        )

        # Size: [batch, sequence, num_heads]
        buckets = self.get_buckets(q)
        
        # imprimimos la distribución de los buckets para ver si es uniforme
        # input(buckets.int().view(-1).bincount()) 

        # buckets = self.adjust_buckets(buckets.int())
        
        # imprimimos la distribución de los buckets para ver si es uniforme
        # print(buckets.int().view(-1).bincount()) 
        

        # buckets = buckets.view(x_size[0] * x_size[1], self.num_heads)

        one_hot_buckets = torch.nn.functional.one_hot(buckets.long(), num_classes=self.n_buckets)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Size: [n_buckets, batch, num_heads, sequence, embedding_dim // num_heads]
        q_masked = torch.einsum('bhne,bnhc->cbhne',q,one_hot_buckets)
        # q_masked = q.masked_fill(masks.unsqueeze(4),0)
        attention = torch.matmul(q_masked,q_masked.transpose(3,4)) / math.sqrt(self.embedding_dim)
        attention = torch.sum(attention, dim=0)
        attention = F.softmax(attention, dim=-1)

        output = torch.matmul(attention, v)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(x_size[0], x_size[1], self.embedding_dim)
        )
        return output

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

class EncoderModelLSH(torch.nn.Module):
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
        self.self_attention = LSHmodule(embedding_dim, num_heads, n_buckets)

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
    
class PytorchModel(torch.nn.Module):
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

        encoder_layers = torch.nn.TransformerEncoderLayer(
            embedding_dim,
            num_heads,
            hidden_size,
            0.2,
            batch_first=True,
            norm_first=False,
            activation="relu",
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, 6)

        # dropout
        self.dropout = torch.nn.Dropout(0.2)

        # classification
        self.model = torch.nn.Linear(embedding_dim * sequence_length, num_classes)

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
        x += self.positional_encodings(x)

        x = self.transformer_encoder(x)

        x = x.view(x.size(0), -1)

        return self.model(x)