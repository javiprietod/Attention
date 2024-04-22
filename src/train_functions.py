# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# own modules
from src.utils import Accuracy, print_confusion_matrix


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the training step.

    Args:
        model: pytorch model.
        train_data: train dataloader.
        loss: loss function.
        optimizer: optimizer object.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    # define metric lists
    losses: list[float] = []
    accuracies: list[float] = []
    model.train()
    accuracy = Accuracy()

    iterator = train_data if len(train_data) < 400 else tqdm(train_data)

    for text, label in iterator:
        accuracy.reset()
        text = text.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        outputs = model(text)
        loss_value = loss(outputs, label)

        loss_value.backward()
        optimizer.step()

        losses.append(loss_value.item())
        accuracy.update(outputs, label)
        accuracies.append(accuracy.compute())

    # write on tensorboard
    if writer is not None:
        writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)
        writer.add_scalar("train/loss", np.mean(losses), epoch)


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the validation step.

    Args:
        model: pytorch model.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """
    model.eval()

    with torch.no_grad():
        # define metric lists
        losses: list[float] = []
        accuracies: list[float] = []

        accuracy = Accuracy()
        for text, label in val_data:
            accuracy.reset()
            text = text.to(device)
            label = label.to(device)

            outputs = model(text)
            loss_value = loss(outputs, label.long())
            losses.append(loss_value.item())
            accuracy.update(outputs, label)
            accuracies.append(accuracy.compute())

        print(
            f"Epoch: {epoch}, Loss: {np.mean(losses)}, Accuracy: {np.mean(accuracies)}"
        )
        # write on tensorboard
        if writer is not None:
            writer.add_scalar("val/accuracy", np.mean(accuracies), epoch)
            writer.add_scalar("val/loss", np.mean(losses), epoch)


def test_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
    int_to_target: dict[int, str],
) -> float:
    """
    This function computes the test step.

    Args:
        model: pytorch model.
        val_data: dataloader of test data.
        device: device of model.

    Returns:
        average accuracy.
    """
    accuracies: list[float] = []
    accuracy = Accuracy()
    confusion_matrix = torch.zeros((len(int_to_target), len(int_to_target)))
    model.eval()

    with torch.no_grad():
        for text, label in test_data:
            text = text.to(device)
            label = label.to(device)

            outputs = model(text)
            accuracy.update(outputs, label)
            accuracies.append(accuracy.compute())
            predictions = outputs.argmax(1).type_as(label)
            for i in range(len(predictions)):
                confusion_matrix[label[i]][predictions[i]] += 1

    print(f"Confusion matrix:")
    print_confusion_matrix(confusion_matrix, int_to_target)

    return np.mean(accuracies, dtype="float")
