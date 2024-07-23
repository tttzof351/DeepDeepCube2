import numpy as np
import random
from tqdm import tqdm
import torch
import time
from numba import njit, jit, types
from numba.typed import List

from hyperparams import hp
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

state_type = types.Array(types.int32, 1, 'C')  # 1D array of float64
scramble_type = types.Tuple((state_type, types.int64, types.int64))

@njit
def _fast_get_scramble(
        initial_state: np.array, 
        action_size: int,
        space_size: int,
        actions: np.array,
        length: int,
): 
    # scramble_states = List.empty_list(state_type)
    scramble_states = np.empty((length, space_size), dtype=np.int32)
    scramble_actions = List.empty_list(types.int64)
    scramble_values = List.empty_list(types.int64)

    state = initial_state

    # dot_hash = np.dot(state, random_vector_cube3)
    # uniq_hashes.add(array_wyhash(game.initial_state))
    i = 0
    while i < length:
        action = np.random.choice(action_size)
        new_state = state[actions[action]].astype(np.int32)
        
        # if len(scramble_states) < 1 or not np.array_equal(new_state, scramble_states[-1]):
        has_duplicates = False
        # for s in scramble_states[-3:]:
        #     has_duplicates = has_duplicates or np.array_equal(s, new_state)

        if not has_duplicates:
            # scramble_states.append(new_state)
            scramble_states[i, :] = new_state
            scramble_actions.append(action)
            scramble_values.append(len(scramble_values) + 1)
            i += 1

        state = new_state
    
    # a = np.stack(scramble_states, axis=0)
    return scramble_states, scramble_actions, scramble_values

def get_fast_scramble(game, length: int): # return (state, action, length)
    return _fast_get_scramble(
        initial_state=game.initial_state,
        action_size=game.action_size,
        space_size=game.space_size,
        actions=game.actions,
        length=length,
    )

@torch.jit.script
def get_torch_scrambles(
    n: int,
    space_size: int,
    action_size: int,
    length: int,
    permutations: torch.Tensor
):
    states = torch.zeros(
        size=(n, length, space_size),
        dtype=torch.int64
    )

    states[:, 0, :] = torch.arange(
        0,
        space_size,
        dtype=torch.int64,
        device=permutations.device
    ).expand(n, space_size)

    actions = torch.randint(
        low=0, 
        high=action_size, 
        size=(n, length),
        device=permutations.device
    )
    
    lengths = torch.arange(
        1, 
        length + 1,
        dtype=torch.float32,
        device=permutations.device
    ).expand(n, length).clone()

    action = actions[:, 0]
    permutation = permutations[action]

    states[:, 0, :] = torch.gather(
        input=states[:, 0, :],
        dim=1,
        index=permutation
    )

    for i in range(1, length):
        action = actions[:, i]
        permutation = permutations[action]

        states[:, i, :] = torch.gather(
            input=states[:, i - 1, :],
            dim=1,
            index=permutation
        )
    
    return states.view(-1, space_size), actions.view(-1), lengths.view(-1)
    # return states, actions, lengths

@torch.jit.script
def reverse_actions(actions: torch.Tensor, n_gens: int):
    n_gens_half = n_gens / 2
    return actions - n_gens_half * (2 * (actions >= n_gens_half).int() - 1)

class Cube3Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            size: int,
            length: int,
            permutations: np.array,
            n: int = 100,
            device = "cpu",
            seed=0
        ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.seed = seed
        self.n = n
        self.device = device
        self.size = size
        
        self.permutations = torch.tensor(
            permutations, 
            dtype=torch.int64,
            device=device,            
        )

        self.action_size = self.permutations.shape[0]
        self.space_size = self.permutations.shape[1]

        self.init_state = torch.arange(
            0, 
            self.permutations.shape[1],             
            device=device
        )
        self.length = length

    @torch.jit.export
    def __getitem__(self, idx):
        return get_torch_scrambles(
            n=self.n,
            space_size=self.space_size,
            action_size=self.action_size,
            length=self.length,
            permutations=self.permutations
        )
        
    def __len__(self):
        return self.size

if __name__ == "__main__":
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    dataset = Cube3Dataset(
        length=26, 
        permutations=game.actions, 
        n=2,
        size=100
    )
    start = time.time()
    states, actions, lengths = next(iter(dataset))
    end = time.time()

    duration = end - start
    print("states:", states.shape)
    print("states:", states[:2, :])