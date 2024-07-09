import os
import sys
import numpy as np
import pybind11
import torch
import pickle as pkl
import time

from catboost import CatBoostRegressor
from accelerate import Accelerator

import random

sys.path.append("./build")
sys.path.append("../")

from cube3_game import Cube3Game
from utils import open_pickle
import cpp_a_star

# with open("../assets/tests/test_states.pickle", "rb") as f:
#     test_states = pkl.load(f)
    
# with open("../assets/tests/test_distance.pickle", "rb") as f:
#     test_distance = pkl.load(f)

game = Cube3Game("../assets/envs/qtm_cube3.pickle")
cpp_a_star.init_envs(game.actions)

validation = open_pickle("../assets/data/validation/val.pickle")
test = validation = open_pickle("../assets/data/test/test.pickle")
deepcube_test = open_pickle("../assets/data/deepcubea/data_0.pkl")


# i = 1

# # v = validation["values"][i]
# # state = validation["states"][i]

# # v = test["values"][i]
# # state = test["states"][i]

# v = len(deepcube_test["solutions"][i])
# state = deepcube_test["states"][i]

# print("catboost_search_a:")
# # result = cpp_a_star.catboost_search_a(
# #     state, # state
# #     10_000_000, # limit size
# #     True # debug
# # )

# result = cpp_a_star.catboost_parallel_search_a(
#     state, # state
#     10_000_000, # limit size
#     True, # debug,
#     10, # parallel_size,
#     100000, # open_max_size
#     1.0 # alpha
# )


# print("i:", i)
# print("V:", v)
# print("Result actions: ", result.actions[1:])
# print("Result size: ", len(result.actions[1:]))
# print("Result h_values: ", [np.round(h, 3) for h in result.h_values])
# print("Result visit_nodes: ", result.visit_nodes)

# # cpp_a_star.run_openmp_test()

cpp_a_star.benchmark_catboost_inference()