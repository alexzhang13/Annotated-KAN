from typing import Dict

# Imported libraries
import torch
from tqdm import tqdm
import numpy as np

# User-defined libraries
from KAN import KAN


def regularization(
    model: KAN,
    l1_factor: float = 1,
    entropy_factor: float = 1,
):
    """
    Regularization described in the original KAN paper. Involves an L1
    and an entropy factor.
    """
    return l1_factor * l1_regularization(
        model
    ) + entropy_factor * entropy_regularization(model)


def l1_regularization(model: KAN):
    """
    Compute L1 regularization of activations by using
    cached activations. Must be called after KAN forward pass
    during training.
    """
    reg = torch.tensor(0.0)
    # regularize coefficient to encourage spline to be zero
    for i in range(len(model.layers)):
        reg += model.layers[i].l1_activations

    return reg


def entropy_regularization(model: KAN):
    """
    Compute entropy regularization of activations by using
    cached activations. Must be called after KAN forward pass
    during training.
    """
    reg = torch.tensor(0.0)
    eps = 1e-4
    # regularize coefficient to encourage spline to be zero
    for i in range(len(model.layers)):
        activations = (
            torch.mean(torch.abs(model.layers[i].l1_activations), dim=0)
            / model.layers[i].l1_activations
        )
        entropy = -torch.sum(activations * torch.log(activations + eps))
        reg += entropy

    return reg


def train(
    model: KAN,
    dataset: Dict[str, torch.Tensor],
    batch_size: int,
    batch_size_test: int,
    device: torch.device,
    reg_lambda: float = 0.1,
    steps: int = 10000,
    loss_fn=None,
    log: int = 20,
    lr: float = 3e-5,
):
    pbar = tqdm(range(steps), desc="KAN Training", ncols=100)

    loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results = {}
    results["train_loss"] = []
    results["test_loss"] = []
    results["reg"] = []
    results["best_test_loss"] = []
    best_test_loss = torch.tensor(1e9)

    for _ in pbar:
        train_id = np.random.choice(
            dataset["train_input"].shape[0], batch_size, replace=False
        )
        test_id = np.random.choice(
            dataset["test_input"].shape[0], batch_size_test, replace=False
        )

        pred = model.forward(dataset["train_input"][train_id].to(device))
        train_loss = loss_fn(pred, dataset["train_label"][train_id].to(device))
        reg_ = regularization(model)
        loss = train_loss + reg_lambda * reg_
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_loss = loss_fn_eval(
            model.forward(dataset["test_input"][test_id].to(device)),
            dataset["test_label"][test_id].to(device),
        )
        if best_test_loss > test_loss:
            best_test_loss = test_loss

        if _ % log == 0:
            pbar.set_description(
                "train loss: %.2e | test loss: %.2e | reg: %.2e "
                % (
                    torch.sqrt(train_loss).cpu().detach().numpy(),
                    torch.sqrt(test_loss).cpu().detach().numpy(),
                    reg_.cpu().detach().numpy(),
                )
            )

        results["train_loss"].append(torch.sqrt(train_loss).cpu().detach().numpy())
        results["test_loss"].append(torch.sqrt(test_loss).cpu().detach().numpy())
        results["best_test_loss"].append(best_test_loss.cpu().detach().numpy())
        results["reg"].append(reg_.cpu().detach().numpy())

        # if save_fig and _ % save_fig_freq == 0:
        #     self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
        #     plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
        #     plt.close()

    return results


if __name__ == "__main__":
    print("KAN Trainer Unit Tests")
