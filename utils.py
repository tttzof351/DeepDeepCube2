import os

import torch
import numpy as np
import random
import wyhash
from numba import njit
import pickle
import time

from catboost import CatBoostRegressor

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

def benchmark_catboost_inference():
    print("os.cpu_count():", os.cpu_count())
    model = CatBoostRegressor()
    model.load_model("./assets/models/catboost_cube3.cb")
    data = np.random.randint(0, 54, size=(100_000, 54))

    start = time.time()
    out = model.predict(data, thread_count=os.cpu_count())
    end = time.time()

    duration = np.round(1000 * (end - start), 3)
    print(f"duration: {duration} ms")

def int_to_human(number):
    if number < 1000:
        return str(number)
    elif number < 1_000_000:
        return f"{np.round(number / 1_000, 3)}K"
    elif number < 1_000_000_000:
        return f"{np.round(number / 1_000_000, 3)}M"
    else:
        return f"{np.round(number / 1_000_000_000, 3)}B"

if __name__ == "__main__":
    # benchmark_catboost_inference()
    print(int_to_human(1010120))
    print(int_to_human(2021))
    print(int_to_human(10_000_010))