# Python Libraries
from typing import List

# Imported Libraries
import numpy as np
import torch

# Helper function derived from https://github.com/KindXiaoming/pykan/blob/master/kan/utils.py
def create_dataset(
    f,
    n_var: int=2,
    ranges=[-1, 1],
    train_num: int =1000,
    test_num: int=1000,
    device: torch.device = torch.device("cpu"),
    seed: int=0,
):
    """
    Create a synthetic dataset as a function of n_var variables
    """
    def normalize(data, mean, std):
        return (data - mean) / std

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)

    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:, i] = (
            torch.rand(
                train_num,
            )
            * (ranges[i, 1] - ranges[i, 0])
            + ranges[i, 0]
        )
        test_input[:, i] = (
            torch.rand(
                test_num,
            )
            * (ranges[i, 1] - ranges[i, 0])
            + ranges[i, 0]
        )

    train_label = f(train_input)
    test_label = f(test_input)

    mean_input = torch.mean(train_input, dim=0, keepdim=True)
    std_input = torch.std(train_input, dim=0, keepdim=True)
    train_input = normalize(train_input, mean_input, std_input)
    test_input = normalize(test_input, mean_input, std_input)

    mean_label = torch.mean(train_label, dim=0, keepdim=True)
    std_label = torch.std(train_label, dim=0, keepdim=True)
    train_label = normalize(train_label, mean_label, std_label)
    test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset["train_input"] = train_input.to(device)
    dataset["test_input"] = test_input.to(device)

    dataset["train_label"] = train_label.to(device)
    dataset["test_label"] = test_label.to(device)

    return dataset
