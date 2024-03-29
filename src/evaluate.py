# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# own modules
from src.utils import (
    load_text_data,
    set_seed,
)
from src.train_functions import test_step

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"


def main(name: str) -> float:
    """
    This function is the main program for the testing.
    """

    # TODO

    # load data
    test_data: DataLoader
    _, _, test_data, _, _, _, _ = load_text_data(
        DATA_PATH, batch_size=int(name.split("_")[-3])
    )

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt").to(device)

    # call test step and evaluate accuracy
    accuracy: float = test_step(model, test_data, device)

    return accuracy


if __name__ == "__main__":
    print(f"accuracy: {main('model_0.0008_16_15_1')}")
