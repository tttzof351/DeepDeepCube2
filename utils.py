import torch
import numpy as np
import random
import wyhash
from numba import njit

sec = wyhash.make_secret(0)

def array_wyhash(array: np.array):
    return wyhash.hash(array.tobytes(), 0, sec)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@njit(cache=True)
def str_hash(array: np.array):
    s = ""
    for e in array:
        s += str(e) + ", "
    return hash(s)

if __name__ == "__main__":
    set_seed(0)
    print(random.randint(0, 100))
    set_seed(0)
    print(random.randint(0, 100))    