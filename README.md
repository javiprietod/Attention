# Attention
This repository is created to investigate and analyse the computational efficiency with attention. The main goal is to compare the computational efficiency of different attention mechanisms in different datasets. 


## How to run the programs
To run the programs, you will need to install the libraries provided in the requirements.txt file. You can do this by running the following command:
``` cmd
pip install -r requirements.txt
```

There are 2 programs that you can run:
 - `train.py`: This program uses the Emotions Dataset to compare the accuracy of different attention mechanisms. In the beginning of the file you can change the model you want to use by changing the `MODEL_NAME` variable with one of the ones provided in the type hint. You can also change the dataset by changing the `DATASET_NAME` variable with one of the ones provided in the type hint but we recommend using the Emotions Dataset. You can run this using the following command:
    ``` cmd
    python -m src.train
    ```

 - `attention_benchmark`: This program uses the IMDB Dataset to compare the computational efficiency of different attention mechanisms. In the end of the file you can change the mechanism you want to use by changing the argument you pass to the main function. You can run this using the following command:
    ``` cmd
    python -m src.attention_benchmark
    ```

## Datasets
The datasets used in this repository are:
1. [Emotions Dataset](https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text):
    This dataset is smaller (68 of maximum sequence length) and has 6 classes. The dataset is used to analyze the accuracy that can be achieved with different attention mechanisms.
2. [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews):
    This dataset is larger (median sequence length of 2472) and has 2 classes. The dataset is used to analyze the computational efficiency and speedup that can be achieved with different attention mechanisms, in time and memory.


