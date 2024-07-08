import os 
import sys
import time 

import numpy as np
import argparse
from tqdm import tqdm
import pickle 

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from numba import njit

from utils import array_wyhash, set_seed
from utils import open_pickle, save_pickle

from cube3_game import Cube3Game
from datasets import get_scramble
from hyperparams import hp

def generate_val_test():
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    set_seed(hp.val_seed)

    scrambles = []
    for _ in tqdm(range(50)):
        scramble = get_scramble(game, hp.cube3_god_number)
        scrambles += scramble
    
    states = np.array([s['state'] for s in scrambles]).astype(np.int32)
    actions = np.array([s['action'] for s in scrambles]).astype(np.int32)
    values = np.array([s['value'] for s in scrambles]).astype(np.int32)

    validation = {
        "states": states,
        "actions": actions,
        "values": values   
    }

    save_pickle(validation, "./assets/data/validation/val.pickle")

    set_seed(hp.test_seed)
    scrambles = []
    for _ in tqdm(range(1000)):
        scramble = get_scramble(game, hp.cube3_test_srambles)
        scrambles += [scramble[-1]]
    
    states = np.array([s['state'] for s in scrambles]).astype(np.int32)
    actions = np.array([s['action'] for s in scrambles]).astype(np.int32)
    values = np.array([s['value'] for s in scrambles]).astype(np.int32)

    test = {
        "states": states,
        "actions": actions,
        "values": values
    }

    save_pickle(test, "./assets/data/test/test.pickle")


def train_catboost():
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")

    validation = open_pickle("./assets/data/validation/val.pickle")

    val_states = validation['states'][:700]
    val_values = validation['values'][:700]

    test_states = validation['states'][700:]
    test_values = validation['values'][700:]

    set_seed(hp.train_seed)
    # for _ in range(10):
    scrambles = []
    for _ in tqdm(range(1_000_000)): # Each scamble contains 26 samples, so trainset 26M
        scramble = get_scramble(game, hp.cube3_god_number)
        scrambles += scramble

    train_states = np.array([s['state'] for s in scrambles]).astype(np.int32)
    train_values = np.array([s['value'] for s in scrambles]).astype(np.int32)

    cb_path_model = f"./assets/models/catboost_cube3.cb"
    init_model = None
    # if os.path.isfile(cb_path_model):
    #     init_model = cb_path_model
    #     print("START FROM!")

    model = CatBoostRegressor(
        verbose=False, 
        iterations=10_000,
        # iterations=1_000,
        max_depth=8,
        use_best_model=True,
        loss_function='RMSE'
    )
    model.fit(
        train_states,
        train_values,
        verbose=True,
        init_model=init_model,
        eval_set=(val_states, val_values)
    )

    test_predictions = model.predict(test_states)
    r2_test = r2_score(test_values, test_predictions)
    rmse_test = root_mean_squared_error(test_values, test_predictions)

    model.save_model(cb_path_model)
    model.save_model(f"./assets/models/catboost_cube3.cpp", format="CPP")

    print("r2_test:", r2_test)
    print("rmse_test:", rmse_test)

    # pass

def test_deepcube():
    sys.path.append("./cpp/build")
    import cpp_a_star

    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    cpp_a_star.init_envs(game.actions)

    deepcube_test = open_pickle("./assets/data/deepcubea/data_0.pkl")
    states = deepcube_test['states']
    opt_solutions = deepcube_test['solutions']

    report = {
        "i": [],
        "states": [],
        "solutions": [],
        "visit_nodes": [],
        "h_values": [], 
        "time_sec": []       
    }

    for i in tqdm(range(len(states))):
        start = time.time()

        state = states[i]
        result = cpp_a_star.catboost_parallel_search_a(
            state, # state
            10_000_000, # limit size
            True, # debug,
            10, # parallel_size,
            100000, # open_max_size
            1.0 # alpha
        )

        end = time.time()
        duration = end - start

        solution = result.actions[1:] if len(result.actions) > 0 else []
        h_values = [np.round(h, 3) for h in result.h_values]

        report["i"].append(i)
        report["states"].append(state)
        report["solutions"].append(solution)
        report["visit_nodes"].append(result.visit_nodes)
        report["h_values"].append(h_values)
        report["time_sec"].append(duration)

        v = len(opt_solutions[i])
        
        print("i:", i)
        print("V:", v)
        print("Result actions: ", solution)
        print("Result size: ", len(solution))
        print("Result h_values: ", [np.round(h, 3) for h in result.h_values])
        print("Result visit_nodes: ", result.visit_nodes)
        print("Time sec:", np.round(duration, 3))

        save_pickle(report, "./assets/reports/result_cb_on_deepcube.pickle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    args = parser.parse_args()

    if args.mode == "generate_val_test":
        generate_val_test()
    elif args.mode == "train_catboost":
        train_catboost()
    elif args.mode == "test_deepcube":
        test_deepcube()