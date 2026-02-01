from collections.abc import Callable, Iterable
from typing import Optional
import torch
import torch.nn as nn
import math
import numpy as np

def data_loader(x, batch_size, context_length, device):
    ix = torch.randint(len(x) - context_length, (batch_size,))
    X = torch.stack([torch.from_numpy(x[i:i+context_length].astype(np.int64)) for i in ix])
    Y = torch.stack([torch.from_numpy(x[i+1:i+context_length+1].astype(np.int64)) for i in ix])
    return X.to(device), Y.to(device)