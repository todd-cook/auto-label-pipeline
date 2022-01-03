"""`utils.py` - data utilities"""
import os
import pathlib
import random
import numpy as np


def fix_path(filepath: str, curr_dir=None) -> str:
    """Correct relative file paths regardless of where execution starts.
    Allows one to use source files in IDEs and DVC pipelines."""
    if not curr_dir:
        curr_dir = pathlib.Path(__file__).parent.resolve()
    new_dir = curr_dir / filepath
    return str(new_dir.resolve())


def seed_everything(seed_value: int):
    """Set random seeds to aim for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    # NOTE: if you copy and use this elsewhere, you'll want to
    # uncomment the sections below as appropriate
    # torch.manual_seed(seed_value)
    # PYTHONHASHSEED should be set before a Python program starts
    # But this applies to forked processes too.
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed_value)
    #     torch.cuda.manual_seed_all(seed_value)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = True
    # tf.random.set_seed(seed_value)
