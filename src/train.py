# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm  # type: ignore
import json

# own modules
from src.models import EncoderModel
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


def main(model_name: str) -> None:
    """
    This function is the main program for the training.
    """

    # empty nohup file
    open("nohup.out", "w").close()

    with open("params.json", "r") as file:
        params = json.load(file)[model_name]

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, test_data, vocab_to_int, _, _, int_to_target = load_text_data(
        DATA_PATH, batch_size=params['batch_size']
    )

    # define name and writer
    name: str = f"model_{params['lr']}_{params['hidden_size']}_{params['batch_size']}_{params['epochs']}_{params['encoders']}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    inputs: torch.Tensor = next(iter(train_data))[0]
    model: torch.nn.Module = EncoderModel(
        sequence_length=inputs.shape[1],
        vocab_to_int=vocab_to_int,
        num_classes=NUMBER_OF_CLASSES,
        **params,
    ).to(device)

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=params['step_size'], gamma=params['gamma'], verbose=True
    )

    # train loop
    for epoch in tqdm(range(params['epochs'])):
        # call train step
        train_step(model, train_data, loss, optimizer, writer, epoch, device)

        # call val step
        val_step(model, val_data, loss, writer, epoch, device)

        # step scheduler
        scheduler.step()

    print(test_step(model, test_data, device, int_to_target))

    # save model
    save_model(model, name)

    return None


if __name__ == "__main__":
    main('EncoderModel')
