import os

import time

import torch
import numpy as np
import random
from numba import njit
import pickle
import time

from catboost import CatBoostRegressor

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

class TimeContext:
    full_time: float = 0.0
    def __init__(self, msg: str, verbose: bool):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        duration = np.round(self.end - self.start, 3)
        TimeContext.full_time += duration
        if self.verbose:
            print(f"{self.msg}: {duration} sec")

def inverse_permutation(perm):
    inverse = [0] * len(perm)
    
    for i, p in enumerate(perm):
        inverse[p] = i
    
    return inverse

if __name__ == "__main__":
    # benchmark_catboost_inference()
    print(int_to_human(1010120))
    print(int_to_human(2021))
    print(int_to_human(10_000_010))