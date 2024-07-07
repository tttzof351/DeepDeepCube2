import os 

import numpy as np
import argparse
from tqdm import tqdm
import pickle 

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from numba import njit

from utils import array_wyhash, set_seed
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

    with open("./assets/data/validation/val.pickle", "wb") as f:
        pickle.dump(validation, f)

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

    with open("./assets/data/test/test.pickle", "wb") as f:
        pickle.dump(test, f)

def train_catboost():
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")

    with open("./assets/data/validation/val.pickle", "rb") as f:
        validation = pickle.load(f)

    val_states = validation['states'][:700]
    val_values = validation['values'][:700]

    test_states = validation['states'][700:]
    test_values = validation['values'][700:]

    set_seed(hp.train_seed)
    # for _ in range(10):
    scrambles = []
    for _ in tqdm(range(1_000_000)):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    args = parser.parse_args()

    if args.mode == "generate_val_test":
        generate_val_test()
    elif args.mode == "train_catboost":
        train_catboost()