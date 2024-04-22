# deep learning libraries
import torch
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from torch.jit import RecursiveScriptModule

# other libraries
import numpy as np
import pandas as pd
import os
import random
import requests
from collections import Counter


class GeneralDataset(Dataset):
    """
    This class for general text datasets used in the project.
    """

    def __init__(
        self,
        path: str,
        vocab_to_int: dict[str, int],
        start_token: str = "<s>",
        end_token: str = "<e>",
        pad_token: str = "<unk>",
        lenght: int = None,
    ) -> None:
        """
        This is the constructor of the GeneralDataset class.

        Args:
            path: path of the dataset.
            vocab_to_int: vocabulary to map words to integers.
            start_token: start token for the text.
            end_token: end token for the text.
            pad_token: padding token for the text.
        """

        super().__init__()
        self.path = path
        self.dataset = pd.read_csv(path)
        self.text_column, self.label_column = list(self.dataset.columns)
        self.tokenizer = get_tokenizer("basic_english")
        self.max_len = max([len(self.tokenizer(text)) for text in self.dataset[self.text_column]])
        if lenght is not None and lenght < self.max_len:
            self.max_len = lenght
        self.vocab_to_int = vocab_to_int
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token

    def __len__(self) -> int:
        """
        This method returns the length of the dataset.

        Returns:
            length of dataset.
        """

        return len(self.dataset)


class EmotionsDataset(GeneralDataset):
    """
    This class is the Emotions Dataset.
    https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text
    """

    def __init__(
        self,
        path: str,
        vocab_to_int: dict[str, int],
        start_token: str = "<s>",
        end_token: str = "<e>",
        pad_token: str = "<unk>",
        lenght: int = None,
    ) -> None:
        super().__init__(path, vocab_to_int, start_token, end_token, pad_token, lenght)

    def __getitem__(self, index: int) -> tuple[list[str], str]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            tuple with text (already padded) and label.
        """

        text = (
            [self.start_token]
            + self.tokenizer(self.dataset[self.text_column][index])[: self.max_len-2]
            + [self.end_token]
        )

        text = [self.pad_token] * (self.max_len - len(text)) + text
        label = self.dataset[self.label_column][index]
        return text, label


class IMBDDataset(GeneralDataset):
    """
    This class is the IMDB reviews dataset.
    https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    """

    def __init__(
        self,
        path: str,
        vocab_to_int: dict[str, int],
        start_token: str = "<s>",
        end_token: str = "<e>",
        pad_token: str = "<unk>",
        lenght: int = None,
    ) -> None:
        super().__init__(path, vocab_to_int, start_token, end_token, pad_token, lenght)

    def __getitem__(self, index: int) -> tuple[list[str], str]:
        """
        This method loads an item based on the index.

        Args:
            index: index of the element in the dataset.

        Returns:
            tuple with text (already padded) and label.
        """

        text = (
            [self.start_token]
            + self.tokenizer(
                self.dataset[self.text_column][index]
            )[: self.max_len-2]
            + [self.end_token]
        )
        text = [self.pad_token] * (self.max_len - len(text)) + text
        label = self.dataset[self.label_column][index]
        return text, label


def collate_fn(
    batch: list[tuple[list[str], str]],
    vocab_to_int: dict[str, int],
    target_to_int: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function creates a batch of text and labels.

    Args:
        batch: list of tuples with text and labels.
        vocab_to_int: vocabulary to map words to integers.
        target_to_int: vocabulary to map labels to integers.

    Returns:
        tuple with text and labels encoded.
    """
    texts: tuple[list[str]]
    labels: tuple[str]
    texts, labels = list(zip(*batch))

    # Encoding words and labels
    texts_encoded = [
        [
            vocab_to_int[word] if word in vocab_to_int else len(vocab_to_int) - 1
            for word in text
        ]
        for text in texts
    ]
    labels_encoded = [target_to_int[label] for label in labels]

    return torch.tensor(texts_encoded, dtype=torch.int), torch.tensor(
        labels_encoded, dtype=torch.long
    )


def create_lookup_tables(
    words: list[str],
    start_token: str = "<s>",
    end_token: str = "<e>",
    unk_token: str = "<unk>",
) -> tuple[dict[str, int], dict[int, str]]:
    """
    This function creates lookup tables for vocabulary.

    Args:
        words: list of words from which to create vocabulary.

    Returns:
        tuple containing two dictionaries:
            vocab_to_int: vocabulary to map words to integers.
            int_to_vocab: vocabulary to map integers to words.
    """
    word_counts = Counter(words)

    sorted_vocab = sorted(word_counts, reverse=True)

    vocab_to_int: dict[str, int] = {word: ii for ii, word in enumerate(sorted_vocab)}
    vocab_to_int[start_token] = len(vocab_to_int)
    vocab_to_int[end_token] = len(vocab_to_int)
    vocab_to_int[unk_token] = len(vocab_to_int)
    int_to_vocab = {ii: word for word, ii in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


def download_google_sheet(sheet_url, output_path):
    # Send a GET request to the sheet's URL
    response = requests.get(sheet_url)

    # Check if the request was successful
    response.raise_for_status()

    # Write the content of the response to a file in the specified output path
    with open(output_path, "wb") as file:
        file.write(response.content)


def load_text_data(
    path: str,
    dataset_name: str,
    batch_size: int = 128,
    num_workers: int = 0,
    length: int = None,
    start_token: str = "<s>",
    end_token: str = "<e>",
    pad_token: str = "<unk>",
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    dict[str, int],
    dict[int, str],
    dict[str, int],
    dict[int, str],
]:
    """
    This function returns two Dataloaders, one for train data and
    other for validation data for text dataset.

    Args:
        path: path of the dataset.
        batch_size: batch size for dataloaders. Default value: 128.and
        num_workers: number of workers for loading data.
            Default value: 0.

    Returns:
        tuple of dataloaders, train, val and test in respective order.
    """
    csv_path = f"{path}/{dataset_name}.csv"
    # download folders if they are not present
    if not os.path.isfile(csv_path):
        # create main dir
        os.makedirs(f"{path}", exist_ok=True)

        # URL of the Google Sheet for direct download as an CSV file
        if dataset_name == "emotions":
            sheet_url = "https://docs.google.com/spreadsheets/d/1JY0Mfh6zxR1q7gcP4Qht4LuOEMxLhghZZxx7BHgcRLQ/export?format=csv"
        elif dataset_name == "imdb":
            sheet_url = "https://docs.google.com/spreadsheets/d/17nSlt_QKx-6e9tb5NKw8qGOZWSsqFaQqbJ7Aedm9ZhY/export?format=csv"
        else:
            raise ValueError("Dataset not found")
        # Specify the path where you want to save the file
        output_path = csv_path

        # Call the function with the URL and the desired output path
        download_google_sheet(sheet_url, output_path)

    # create lookup tables
    dataset = pd.read_csv(csv_path)
    text_column, label_column = list(dataset.columns)
    words = [
        word
        for text in dataset[text_column]
        for word in get_tokenizer("basic_english")(text)
    ]
    vocab_to_int, int_to_vocab = create_lookup_tables(
        words, start_token, end_token, pad_token
    )
    targets: list[str] = list(dataset[label_column].unique())
    targets_to_int = {target: ii for ii, target in enumerate(targets)}
    int_to_targets = {ii: target for target, ii in targets_to_int.items()}

    train_dataset: Dataset

    # create datasets
    if dataset_name == "emotions":
        train_dataset = EmotionsDataset(
            csv_path, vocab_to_int, start_token, end_token, pad_token, length
        )
    elif dataset_name == "imdb":
        train_dataset = IMBDDataset(
            csv_path, vocab_to_int, start_token, end_token, pad_token, length
        )
    else:
        raise ValueError("Dataset class not found")

    val_dataset: Dataset
    test_dataset: Dataset
    train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2])
    train_dataset, val_dataset = random_split(train_dataset, [0.75, 0.25])

    # define dataloaders
    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, vocab_to_int, targets_to_int),
    )
    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, vocab_to_int, targets_to_int),
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, vocab_to_int, targets_to_int),
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        vocab_to_int,
        int_to_vocab,
        targets_to_int,
        int_to_targets,
    )


def load_benchmark_data(
    path: str,
    batch_size: int = 128,
    percent: float = 1,
    num_workers: int = 0,
    length: int = None,
    start_token: str = "<s>",
    end_token: str = "<e>",
    pad_token: str = "<unk>",
) -> tuple[DataLoader, dict[str, int], dict[int, str], dict[str, int], dict[int, str],]:
    """
    This function returns two Dataloaders, one for train data and
    other for validation data for text dataset.

    Args:
        path: path of the dataset.
        batch_size: batch size for dataloaders. Default value: 128.and
        num_workers: number of workers for loading data.
            Default value: 0.

    Returns:
        tuple of dataloaders, train, val and test in respective order.
    """
    csv_path = f"{path}/imdb.csv"
    # download folders if they are not present
    if not os.path.isfile(csv_path):
        # create main dir
        os.makedirs(f"{path}", exist_ok=True)

        # URL of the Google Sheet for direct download as an CSV file
        sheet_url = """
        https://docs.google.com/spreadsheets/d/17nSlt_QKx-6e9tb5NKw8qGOZWSsqFaQqbJ7Aedm9ZhY/export?format=csv"""

        # Specify the path where you want to save the file
        output_path = csv_path

        # Call the function with the URL and the desired output path
        download_google_sheet(sheet_url, output_path)

    # create lookup tables
    dataset = pd.read_csv(csv_path)
    text_column, label_column = list(dataset.columns)
    words = [
        word
        for text in dataset[text_column]
        for word in get_tokenizer("basic_english")(text)
    ]
    vocab_to_int, int_to_vocab = create_lookup_tables(
        words, start_token, end_token, pad_token
    )
    targets: list[str] = list(dataset[label_column].unique())
    targets_to_int = {target: ii for ii, target in enumerate(targets)}
    int_to_targets = {ii: target for target, ii in targets_to_int.items()}

    # create datasets
    train_dataset: Dataset = IMBDDataset(
        csv_path, vocab_to_int, start_token, end_token, pad_token, length
    )

    assert (
        percent > 0 and percent < 1
    ), "The percentage of train data must be between 0 and 1."

    train_dataset, _ = random_split(train_dataset, [percent, 1 - percent])

    # define dataloaders
    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: collate_fn(x, vocab_to_int, targets_to_int),
    )

    return (
        train_dataloader,
        vocab_to_int,
        int_to_vocab,
        targets_to_int,
        int_to_targets,
    )


class Accuracy:
    """
    This class is the accuracy object.

    Attributes:
        correct: number of correct predictions.
        total: number of total examples to classify.
    """

    correct: int
    total: int

    def __init__(self) -> None:
        """
        This is the constructor of Accuracy class. It should
        initialize correct and total to zero.
        """

        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        This method update the value of correct and total counts.

        Args:
            logits: outputs of the model.
                Dimensions: [batch, number of classes]
            labels: labels of the examples. Dimensions: [batch].
        """

        # compute predictions
        predictions = logits.argmax(1).type_as(labels)

        # update counts
        self.correct += int(predictions.eq(labels).sum().item())
        self.total += labels.shape[0]

        return None

    def compute(self) -> float:
        """
        This method returns the accuracy value.

        Returns:
            accuracy value.
        """

        return self.correct / self.total

    def reset(self) -> None:
        """
        This method resets to zero the count of correct and total number of
        examples.
        """

        # init to zero the counts
        self.correct = 0
        self.total = 0

        return None


def print_confusion_matrix(
    confusion_matrix: torch.Tensor,
    int_to_target: dict[int, str],
):
    """
    Code for printing the confusion matrix.

    Args:
        model: pytorch model.
        val_data: dataloader of test data.
        device: device of model.

    """
    line: str = ""
    line += "\t\u0020"
    for i in range(confusion_matrix.shape[0]):
        line += "   " + int_to_target[i]
    print(line)

    for i in range(confusion_matrix.shape[0]):
        line = ""
        tabs = 1 if i == 3 else 2
        line += int_to_target[i] + "\t" * tabs
        for j in list(int(j.item()) for j in confusion_matrix[i]):
            line += str(j) + "\t"
        print(line)


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
