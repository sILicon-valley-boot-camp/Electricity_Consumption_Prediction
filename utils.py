import os
import torch
import random
import numpy as np


def seed_everything(seed: int):
    """
    Sets the seed for generating random numbers to a fixed value. This is done for all libraries used in the code 
    to ensure the results are reproducible across different runs.

    Args:
        seed (int): Seed value for random number generators.
    """

    # Set the random seed for Python's random library.
    random.seed(seed)

    # Set the random seed for Python's hash function used for string keys in dictionaries.
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set the random seed for NumPy's random library.
    np.random.seed(seed)

    # Set the random seed for PyTorch's CPU random number generator.
    torch.manual_seed(seed)

    # Set the random seed for PyTorch's GPU random number generator.
    torch.cuda.manual_seed(seed)

    # Set the environment variable for cuBLAS library. This determines the maximum workspace size for cuBLAS functions.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Make PyTorch use deterministic algorithms (where available), to ensure consistent results on every run.
    torch.use_deterministic_algorithms(True)

    # Enable the benchmark mode in cudnn. It leads to potential deterministic behavior (trade-off on speed)
    torch.backends.cudnn.benchmark = True
