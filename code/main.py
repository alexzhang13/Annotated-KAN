# Python libraries
import warnings
import random

# Installed libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
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

    f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 3)
    dataset = create_dataset(f, n_var=2, train_num=1000, test_num=100)

    config = KANConfig()
    layer_widths = [2, 1, 1]
    model = KAN(layer_widths, config)

    results = train(
        model,
        dataset=dataset,
        steps=50000,
        batch_size=128,
        batch_size_test=32,
        lr=0.01,
        device=config.device,
    )
    plot_results(results)
    model(dataset["train_input"])
    plot(model)
    model.grid_extension(dataset["train_input"], new_grid_size=50)
    model(dataset["train_input"])
    plot(model)


def split_torch_dataset(train_data, test_data):
    """
    Quick function for splitting dataset into format used
    in rest of notebook. Don't do this for your own code.
    """
    dataset = {}
    dataset['train_input'] = []
    dataset['train_label'] = []
    dataset['test_input'] = []
    dataset['test_label'] = []

    for (x,y) in train_data:
        dataset['train_input'].append(x.flatten()) 
        dataset['train_label'].append(y)

    dataset['train_input'] = torch.stack(dataset['train_input']).squeeze()
    dataset['train_label'] = torch.tensor(dataset['train_label'])
    dataset['train_label'] = F.one_hot(dataset['train_label'], num_classes=10).float()

    for (x,y) in test_data:
        dataset['test_input'].append(x.flatten()) 
        dataset['test_label'].append(y)

    dataset['test_input'] = torch.stack(dataset['test_input']).squeeze()
    dataset['test_label'] = torch.tensor(dataset['test_label'])
    dataset['test_label'] = F.one_hot(dataset['test_label'], num_classes=10).float()

    print('train input size', dataset['train_input'].shape)
    print('train label size', dataset['train_label'].shape)
    print('test input size', dataset['test_input'].shape)
    print('test label size', dataset['test_label'].shape)

    return dataset

def train_mnist():
    """
    Simple training example for MNIST dataset.
    """

    config = KANConfig()
    config.grid_size = 10
    layer_widths = [28 * 28, 64, 10]
    model = KAN(layer_widths, config)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = datasets.MNIST("./data", train=True, download=False, transform=transform)
    test_data = datasets.MNIST("./data", train=False, transform=transform)

    dataset = split_torch_dataset(train_data, test_data)
    loss = nn.BCEWithLogitsLoss()

    results = train(
        model,
        dataset=dataset,
        steps=500,
        batch_size=256,
        batch_size_test=100,
        lr=0.1,
        log=1,
        device=config.device,
        loss_fn=lambda x, y: loss(x, y),
        loss_fn_eval=lambda x, y: (torch.argmax(x, dim=-1) != torch.argmax(y, dim=-1)).sum()
    )
    plot_results(results)


if __name__ == "__main__":
    simple_train()
