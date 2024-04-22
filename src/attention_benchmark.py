# deep learning libraries
import torch
from torch.utils.data import DataLoader

from time import perf_counter
from tqdm.auto import tqdm
from memory_profiler import profile

# own modules
from src.utils import (
    load_benchmark_data,
    set_seed,
)
from src.models import (
    SelfAttention,
    PositionalEncoding,
    KernelizedAttention,
    LocalAttention,
    LocalAttentionUnFold,
    LinformerSelfAttention,
    KernelizedLinformerAttention,
    LSHmodule,
)

# set device
"""
device: torch.device = (
    torch.device("mps") if torch.backends.mps.is_available() 
    else torch.device("cuda") if torch.cuda.is_available() 
    else torch.device("cpu")
)
"""
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"

EMBEDDING_DIM: int = 128
BATCH_SIZE: int = 16
LENGTH: int = None # If None, the length of the longest sentence is used
NUM_CLASSES: int = 2


class AttentionModel(torch.nn.Module):
    """
    This class is the model for the attention models.
    """

    def __init__(
        self,
        attention: torch.nn.Module,
        input_channels: int,
        vocab_to_int: dict[str, int],
    ) -> None:
        super().__init__()
        self.attention = attention
        self.vocab_to_int = vocab_to_int
        self.base_model = torch.nn.Sequential(
            torch.nn.Embedding(len(vocab_to_int), EMBEDDING_DIM, len(vocab_to_int) - 1),
            PositionalEncoding(EMBEDDING_DIM),
        )
        self.linear = torch.nn.Linear(EMBEDDING_DIM * input_channels, NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.base_model(inputs)
        try:
            x = self.attention(x)
        except:
            # multihead attention
            x, _ = self.attention(x, x, x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


def train_pass(
    model: torch.nn.Module,
    train_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss: torch.nn.Module,
    device: torch.device,
):
    """
    This function is used to perform a forward pass the model.
    """
    model.train()

    for text, label in tqdm(train_data):
        text = text.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        outputs = model(text)
        loss_value = loss(outputs, label)

        loss_value.backward()
        optimizer.step()


@torch.no_grad()
def test_pass(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
):
    """
    This function is used to perform a forward pass the model.
    """
    model.eval()

    for text, label in tqdm(test_data):
        text = text.to(device)
        label = label.to(device)

        outputs = model(text)


@profile
def train_benchmark(model: torch.nn.Module, data: DataLoader) -> float:
    """
    This function is the benchmark for the attention models in training
    """

    lr: float = 6e-4

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-4
    )

    s: float = perf_counter()
    train_pass(model, data, optimizer, loss, device)
    train: float = perf_counter() - s

    return train


@profile
def test_benchmark(model: torch.nn.Module, data: DataLoader) -> float:
    """
    This function is the benchmark for the attention models in inference
    """

    s: float = perf_counter()
    test_pass(model, data, device)
    test: float = perf_counter() - s

    return test


def main(
    attention: torch.nn.Module, data: DataLoader, vocab_to_int: dict[str, int]
) -> None:
    """
    This function is the main program for all the benchmarks
    """

    inputs: torch.Tensor = next(iter(data))[0]
    model = AttentionModel(attention, inputs.shape[1], vocab_to_int).to(device)
    train_time = train_benchmark(model, data)
    test_time = test_benchmark(model, data)
    model_name = attention.__class__.__name__

    print("-" * 50)
    print(f"Train time for {model_name}: {train_time}")
    print("-" * 50)
    print(f"Test time for {model_name}: {test_time}")
    print("-" * 50)


if __name__ == "__main__":
    # load data
    data: DataLoader
    vocab_to_int: dict[str, int]
    data, vocab_to_int, _, __, _ = load_benchmark_data(
        DATA_PATH,
        batch_size=BATCH_SIZE,
        percent=0.005,
        length=LENGTH,
    )
    sequence_length: torch.Tensor = next(iter(data))[0].shape[1]

    main(
        SelfAttention(EMBEDDING_DIM, 4),
        # KernelizedAttention(EMBEDDING_DIM, 4, 0),
        # LocalAttention(EMBEDDING_DIM, 4, 7),
        # LocalAttentionUnFold(EMBEDDING_DIM, 4, 7, sequence_length),
        # torch.nn.MultiheadAttention(EMBEDDING_DIM, 4),
        # LinformerSelfAttention(EMBEDDING_DIM, sequence_length, 4),
        # KernelizedLinformerAttention(EMBEDDING_DIM, num_heads=4, mapping_dim=EMBEDDING_DIM//2, seq_len=sequence_length),
        # LSHmodule(EMBEDDING_DIM, 4, 4),
        data=data,
        vocab_to_int=vocab_to_int,
    )
