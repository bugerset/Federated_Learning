import os
import random
import numpy as np
import torch

def set_seed(seed = 845, deterministic: bool = True):
    # Set dictionary seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set random seed
    random.seed(seed)

    # Set numpy random seed
    np.random.seed(seed)

    # Set cpu random seed
    torch.manual_seed(seed)

    # Set gpu(cuda) random seed
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False