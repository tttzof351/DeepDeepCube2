import numpy as np
import argparse
from tqdm import tqdm
import pickle 

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test')
    args = parser.parse_args()

    if args.mode == "generate_val_test":
        generate_val_test()