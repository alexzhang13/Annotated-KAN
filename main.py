# Python libraries
import warnings
import random

# Installed libraries
import torch
import numpy as np


# User defined libraries
from KAN import KAN, KANConfig
from KANTrainer import train
from datasets import create_dataset
from plot import plot, plot_results

warnings.filterwarnings("ignore")


def simple_train():
    """
    Simple training example
    """
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    f = lambda x: (x[:, [0]] ** 3 + x[:, [1]] ** 2)
    dataset = create_dataset(f, n_var=2, train_num=1000, test_num=100)

    config = KANConfig()
    layer_widths = [2, 1, 1]
    model = KAN(layer_widths, config)

    results = train(
        model,
        dataset=dataset,
        steps=10000,
        batch_size=32,
        batch_size_test=8,
        lr=0.01,
        device=config.device,
    )
    plot_results(results)
    model(dataset["train_input"])
    plot(model)
    model.grid_extension(dataset["train_input"], new_grid_size=50)
    model(dataset["train_input"])
    plot(model)


if __name__ == "__main__":
    simple_train()
