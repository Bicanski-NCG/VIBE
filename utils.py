import random
import wandb
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_model_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"model/total_params": total_params})


def save_initial_state(model, path="initial_model.pt", random_path="initial_random_state.pt"):
    torch.save(model.state_dict(), path)
    random_state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(random_state, random_path)


def load_initial_state(model, path="initial_model.pt", random_path="initial_random_state.pt"):
    model.load_state_dict(torch.load(path))
    random_state = torch.load(random_path, weights_only=False)
    random.setstate(random_state["random"])
    np.random.set_state(random_state["numpy"])
    torch.set_rng_state(random_state["torch"])
    if torch.cuda.is_available() and random_state["cuda"] is not None:
        torch.cuda.set_rng_state_all(random_state["cuda"])