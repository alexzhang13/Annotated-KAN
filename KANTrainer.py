from typing import Dict, Optional
import os

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
        acts = model.layers[i].activations
        l1_activations = torch.sum(torch.mean(torch.abs(acts), dim=0))
        reg += l1_activations

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
        acts = model.layers[i].activations
        l1_activations = torch.sum(torch.mean(torch.abs(acts), dim=0))
        activations = (
            torch.mean(torch.abs(l1_activations), dim=0)
            / l1_activations
        )
        entropy = -torch.sum(activations * torch.log(activations + eps))
        reg += entropy

    return reg


# Adapted from https://github.com/KindXiaoming/pykan
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
    # grid_extension_freq: int = 100000,
    # grid_extension_factor: int = 5,
    lr: float = 3e-5,
    save_path: str ='./saved_models/',
    ckpt_name: Optional[str] = 'best.pt',
):
    """
    Train loop for KANs. Logs loss every {log} steps and uses
    the best checkpoint as the trained model. Returns a dict of
    the loss trajectory.
    """
    if not os.path.exists(save_path):
       os.makedirs(save_path) 

    pbar = tqdm(range(steps), desc="KAN Training", ncols=200)

    loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results = {}
    results["train_loss"] = []
    results["test_loss"] = []
    results["regularization"] = []
    results["best_test_loss"] = []

    train_size = dataset["train_input"].shape[0]
    test_size = dataset["test_input"].shape[0]

    best_test_loss = torch.tensor(1e9)

    for step in pbar:
        train_id = np.random.choice(train_size, batch_size, replace=False)
        test_id = np.random.choice(test_size, batch_size_test, replace=False)
        x = dataset["train_input"][train_id].to(device)
        y = dataset["train_label"][train_id].to(device)
        x_eval = dataset["test_input"][test_id].to(device)
        y_eval = dataset["test_input"][test_id].to(device)

        pred = model.forward(x)
        train_loss = loss_fn(pred, y)
        ent_l1_reg = regularization(model)
        loss = train_loss + reg_lambda * ent_l1_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_loss = loss_fn_eval(model.forward(x_eval), y_eval)
        if best_test_loss > test_loss:
            best_test_loss = test_loss
            if ckpt_name is not None:
                print('saving at step', step)
                torch.save(model.state_dict(), os.path.join(save_path, ckpt_name))

        if step % log == 0:
            pbar.set_description(
                "train loss: %.2e | test loss: %.2e | reg: %.2e "
                % (
                    train_loss.cpu().detach().numpy(),
                    test_loss.cpu().detach().numpy(),
                    ent_l1_reg.cpu().detach().numpy(),
                )
            )

        results["train_loss"].append(train_loss.cpu().detach().numpy())
        results["test_loss"].append(test_loss.cpu().detach().numpy())
        results["best_test_loss"].append(best_test_loss.cpu().detach().numpy())
        results["regularization"].append(ent_l1_reg.cpu().detach().numpy())

        # if step % grid_extension_freq == 0:
        #     model.grid_extension(x, model.config.grid_size + grid_extension_factor)

    if ckpt_name is not None:
        model.load_state_dict(torch.load(os.path.join(save_path, ckpt_name)))

    return results


if __name__ == "__main__":
    print("KAN Trainer Unit Tests")
