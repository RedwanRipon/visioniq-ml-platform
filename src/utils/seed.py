import random
import numpy as np
import torch


def set_seed(seed=42):
    """Make Python, NumPy and PyTorch produce the same random numbers every run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)