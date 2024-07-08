import torch
import numpy as np
import random
import wyhash
from numba import njit
import pickle

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

def open_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(data, path: str):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def check_solution(game, state, solution):
    if len(solution) == 0:
        return False
    
    if solution[0] == -1:
        solution = solution[1:]

    for action in solution:
        state = game.apply_action(state, action)

    return game.is_goal(state)    

if __name__ == "__main__":
    set_seed(0)
    print(random.randint(0, 100))
    set_seed(0)
    print(random.randint(0, 100))    