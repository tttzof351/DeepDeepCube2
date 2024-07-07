import numpy as np
import random
from tqdm import tqdm
from cube3_game import Cube3Game
from utils import array_wyhash

def get_scramble(game, length: int): # return (state, action, length)
    scramble = []
    uniq_hashes = set()

    state = game.initial_state

    uniq_hashes.add(array_wyhash(game.initial_state))

    while len(scramble) < length:
        action = np.random.choice(game.action_size)
        new_state = game.apply_action(state, action)
        
        new_state_hash = array_wyhash(new_state)

        if new_state_hash not in uniq_hashes:
            scramble.append({
                "state": new_state,
                "action": action,
                "value": len(scramble) + 1
            })
            state = new_state
            uniq_hashes.add(new_state_hash)
    
    return scramble

if __name__ == "__main__":
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    
    for _ in tqdm(range(100_000)):
        scramble = get_scramble(game, 1000)

    print(len(scramble))