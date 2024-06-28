# Python libraries
import warnings
import random

# Installed libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# User defined libraries
from KAN import KAN, KANConfig
from KANTrainer import train
from datasets import create_dataset
from plot import plot, plot_results

warnings.filterwarnings("ignore")

def main():
    """
    Simple training example
    """
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    f = lambda x: (x[:,[0]]**3 + x[:,[1]]**2)
    dataset = create_dataset(f, n_var=2, train_num=1000, test_num=100)
    dataset['train_input'].shape

    config = KANConfig()
    layer_widths = [2, 1, 1]
    model = KAN(layer_widths, config)
    
    results = train(model, dataset=dataset, steps=50000, batch_size=32, batch_size_test=8, device=config.device)
    # plot_results(results)
    plot(model)


if __name__ == "__main__":
    main()
