# Imported libraries
import matplotlib.pyplot as plt
import torch

# User-defined libraries
from KAN import KAN, KANConfig

def plot():
    """
    Function for plotting the interior of a KAN, similar to the original paper.
    """
    pass


if __name__ == "__main__":
    layer_widths = [1,4,1]
    config = KANConfig()
    model = KAN(layer_widths, config)
