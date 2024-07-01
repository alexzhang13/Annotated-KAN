# Python libraries
import os

# Imported libraries
import matplotlib.pyplot as plt
import torch
import numpy as np

# User-defined libraries
from KAN import KAN, KANConfig


def plot(model: KAN, folder="./figures", scale=0.5, title=None):
    """
    Function for plotting KANs and visualizing their activations adapted from
    https://github.com/KindXiaoming/pykan/blob/master/kan/KAN.py#L561
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    depth = len(model.layer_widths) - 1
    for l in range(depth):
        w_large = 2.0
        for i in range(model.layer_widths[l]):
            for j in range(model.layer_widths[l + 1]):
                rank = torch.argsort(model.layers[l].inp[:, i])
                fig, ax = plt.subplots(figsize=(w_large, w_large))
                plt.gca().patch.set_edgecolor("white")
                plt.gca().patch.set_linewidth(1.5)

                color = "black"
                plt.plot(
                    model.layers[l].inp[:, i][rank].cpu().detach().numpy(),
                    model.layers[l].activations[:, j, i][rank].cpu().detach().numpy(),
                    color=color,
                    lw=5,
                )
                plt.gca().spines[:].set_color(color)
                plt.savefig(
                    f"{folder}/sp_{l}_{i}_{j}.png", bbox_inches="tight", dpi=400
                )
                plt.close()

    # draw skeleton
    width = np.array(model.layer_widths)
    A = 1
    y0 = 0.4  # 0.4

    # plt.figure(figsize=(5,5*(neuron_depth-1)*y0))
    neuron_depth = len(width)
    min_spacing = A / np.maximum(np.max(width), 5)

    max_num_weights = np.max(width[:-1] * width[1:])
    y1 = 0.4 / np.maximum(max_num_weights, 3)

    fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale * (neuron_depth - 1) * y0))

    # plot scatters and lines
    for l in range(neuron_depth):
        n = width[l]
        for i in range(n):
            plt.scatter(
                1 / (2 * n) + i / n,
                l * y0,
                s=min_spacing**2 * 10000 * scale**2,
                color="black",
            )

            if l < neuron_depth - 1:
                # plot connections
                n_next = width[l + 1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j
                    color = "black"
                    plt.plot(
                        [1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N],
                        [l * y0, (l + 1 / 2) * y0 - y1],
                        color=color,
                        lw=2 * scale,
                    )  # alpha=alpha[l][j][i] * alpha_mask)
                    plt.plot(
                        [1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next],
                        [(l + 1 / 2) * y0 + y1, (l + 1) * y0],
                        color=color,
                        lw=2 * scale,
                    )  # alpha=alpha[l][j][i] * alpha_mask)

        plt.xlim(0, 1)
        plt.ylim(-0.1 * y0, (neuron_depth - 1 + 0.1) * y0)

    # -- Transformation functions
    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    # -- Take data coordinates and transform them to normalized figure coordinates
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

    plt.axis("off")

    # plot splines
    for l in range(neuron_depth - 1):
        n = width[l]
        for i in range(n):
            n_next = width[l + 1]
            N = n * n_next
            for j in range(n_next):
                id_ = i * n_next + j
                im = plt.imread(f"{folder}/sp_{l}_{i}_{j}.png")
                left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                bottom = DC_to_NFC([0, (l + 1 / 2) * y0 - y1])[1]
                up = DC_to_NFC([0, (l + 1 / 2) * y0 + y1])[1]
                newax = fig.add_axes((left, bottom, right - left, up - bottom))
                newax.imshow(im)
                newax.axis("off")

    if title is not None:
        plt.title(title)

    plt.show()


def plot_results(results):
    """
    Function for plotting the interior of a KAN, similar to the original paper.
    """
    for key, value in results.items():
        plt.plot(value)
        plt.title(key)
        plt.show()
