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
from utils import set_seed

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
def get_torch_scrambles_2(
    n: int,        
    N: int,
    generators: torch.Tensor,
    hash_vec: torch.Tensor,
):
    assert N <= 10
    assert n <= 26    
    n_gens = generators.shape[0]
    state_size = generators.shape[1]

    states = torch.arange(0, state_size, dtype=torch.int64).unsqueeze(
        dim=0
    ).unsqueeze(
        dim=0
    ).expand(
        N, 1, state_size
    ) # (N, n, STATE_SIZE) == [S1, S1, ..., SN, SN]

    hashes = torch.einsum('NnS,S->Nn', states, hash_vec) # (N * n) == [HASH(S1), HASH(S1), ..., HASH(SN), HASH(SN)]

    i = 0
    actions = torch.full(
        size=[N, 1], 
        fill_value=-1, 
        # dtype=torch.int64
    )

    while i < n:
        action = torch.randint(low=0, high=n_gens, size=(N,))
        
        new_states = torch.gather(
            input=states[:, -1, :],
            dim=1,
            index=generators[action]
        )
        
        new_hashes = torch.einsum('NS,S->N', new_states, hash_vec)
        
        check_hashes = torch.eq(
            hashes, 
            new_hashes.unsqueeze(dim=1).expand(N, hashes.shape[1])
        )
        has_duplicates = check_hashes.any().item()
        if has_duplicates:
            # print("Duplicated!")
            continue
        
        states = torch.cat([states, new_states.unsqueeze(dim=1)], dim=1)
        hashes = torch.cat([hashes, new_hashes.unsqueeze(dim=1)], dim=1)
        actions = torch.cat([actions, action.unsqueeze(dim=1)], dim=1)
        # print("actions:", actions.shape)

        i += 1

        # print("new_states:", new_states.shape)
        # print("new_hashes:", new_hashes.shape)
        # print("check_hashes:", check_hashes)
        # break

    lengths = torch.arange(
        1, 
        n + 1,
        dtype=torch.float32
    ).expand(N, n).clone()

    states = states[:, 1:, :].reshape(
        N * n,
        state_size
    )

    actions = actions[:, 1:].reshape(N * n)

    lengths = lengths.reshape(-1)
    
    return states, actions, lengths


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
            device = "cpu"        
        ):
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

class Cube3Dataset2(torch.utils.data.Dataset):
    def __init__(
            self, 
            n: int,
            N: int,
            size: int,
            generators: torch.Tensor,
            seed=0
        ):
        set_seed(seed)
        
        self.seed = seed
        self.n = n
        self.N = N
        self.state_size = generators.shape[1]
        self.generators = generators
        self.hash_vec = torch.randint(0, 1_000_000_000_000, (self.state_size,))  

        self.size = size

    @torch.jit.export
    def __getitem__(self, idx):
        # print(f"idx: {idx}, n: {self.n}; N: {self.N}")
        return get_torch_scrambles_2(
            N=self.N,
            n=self.n,
            generators=self.generators,
            hash_vec=self.hash_vec
        )
        
    def __len__(self):
        return self.size

if __name__ == "__main__":
    set_seed(0)
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    state_size = game.actions.shape[1]
    hash_vec = torch.randint(0, 1_000_000_000_000, (state_size,))  
    generators = torch.tensor(game.actions, dtype=torch.int64)
    
    N = 10
    n = 26
    
    start = time.time()
    scrambles, actions, lengths = get_torch_scrambles_2(
        N=N,
        n=n,
        generators=generators,
        hash_vec=hash_vec
    )
    end = time.time()

    N_i = 0
    n_i = 0
    
    state = scrambles[N_i * n + n_i]
    s = torch.arange(0, state_size)
    for i in range(n_i + 1):
        a = actions[N_i * n + i]
        # print("a:", a)
        s = s[generators[a]]
        # print("new_state:", s[:10])
    
    # print("s:", s)
    # print("state:", state)
    print("K:", lengths[N_i * n + n_i].item(), "; is_equal?", lengths[N_i * n + n_i].item() == n_i + 1)
    print("is_equal:", (s == state).all().item())
    
    print("scrambles:", scrambles.shape)
    print("actions:", actions.shape)
    print("lengths:", lengths.shape)

    duration = np.round(end - start, 3)
    print(f"Duration: {duration} sec")

    # print("lengths:", lengths)

    # game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    # dataset = Cube3Dataset(
    #     length=26, 
    #     permutations=game.actions, 
    #     n=2,
    #     size=100
    # )
    # start = time.time()
    # states, actions, lengths = next(iter(dataset))
    # end = time.time()

    # duration = end - start
    # print("states:", states.shape)
    # print("states:", states[:2, :])