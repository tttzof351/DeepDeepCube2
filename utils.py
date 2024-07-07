import torch
import numpy as np
import random
import wyhash


sec = wyhash.make_secret(0)

def array_wyhash(array: np.array):
    return wyhash.hash(array.tobytes(), 0, sec)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    set_seed(0)
    print(random.randint(0, 100))
    set_seed(0)
    print(random.randint(0, 100))    