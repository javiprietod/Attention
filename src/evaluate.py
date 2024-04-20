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
    # load data
    test_data: DataLoader
    _, _, test_data, _, _, _, int_to_target = load_text_data(
        DATA_PATH, name.split('_')[1], batch_size=int(name.split("_")[-9])
    )

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt").to(device)

    # call test step and evaluate accuracy
    accuracy: float = test_step(model, test_data, device, int_to_target)

    return accuracy


if __name__ == "__main__":
    print(f"accuracy: {main('KernelizedModel_emotions_lr_0.0006_batch_16_hidden_128_encoders_2_embedding_128_heads_4')}")
