# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm  # type: ignore

# own modules
from src.models import EncoderModel, PytorchModel
from src.utils import (
    load_text_data,
    save_model,
    set_seed,
)
from src.train_functions import train_step, val_step, test_step

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(4)

# static variables
DATA_PATH: str = "data"

NUMBER_OF_CLASSES: int = 6


def main() -> None:
    """
    This function is the main program for the training.
    """

    epochs: int = 15
    lr: float = 6e-4
    batch_size: int = 16
    hidden_size: int = 1024
    embedding_dim: int = 128
    encoders: int = 2
    nhead: int = 4

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, test_data, vocab_to_int, _, _, int_to_target = load_text_data(
        DATA_PATH, batch_size=batch_size
    )

    # define name and writer
    name: str = f"model_{lr}_{hidden_size}_{batch_size}_{epochs}_{encoders}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    inputs: torch.Tensor = next(iter(train_data))[0]
    model: torch.nn.Module = EncoderModel(
        hidden_size=hidden_size,
        vocab_to_int=vocab_to_int,
        input_channels=inputs.shape[1],
        output_channels=NUMBER_OF_CLASSES,
        embedding_dim=embedding_dim,
        encoders=encoders,
        nhead=nhead,
    ).to(device)

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=12, gamma=0.4, verbose=True
    )

    # train loop
    for epoch in tqdm(range(epochs)):
        # call train step
        train_step(model, train_data, loss, optimizer, writer, epoch, device)

        # call val step
        val_step(model, val_data, loss, writer, epoch, device)

        # step scheduler
        scheduler.step()

    # save model
    save_model(model, name)

    print(test_step(model, test_data, device, int_to_target))

    return None


if __name__ == "__main__":
    main()
