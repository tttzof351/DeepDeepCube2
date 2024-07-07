import numpy as np
import wyhash

sec = wyhash.make_secret(0)

def array_wyhash(array: np.array):
    return wyhash.hash(array.tobytes(), 0, sec)