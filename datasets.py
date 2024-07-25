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
def get_torch_scrambles_3(
    n: int,        
    N: int,
    generators: torch.Tensor,
    hash_vec: torch.Tensor,
    device: torch.device
):
    n_gens = generators.shape[0]
    state_size = generators.shape[1]

    states = torch.arange(0, state_size, dtype=torch.int64, device=device).unsqueeze(
        dim=0
    ).unsqueeze(
        dim=0
    ).expand(
        N, 1, state_size
    ) # (N, n=1, STATE_SIZE) == [S1, S1, ..., SN, SN]

    # hashes = torch.einsum(
    #     'NnS,S->Nn', # 
    #     states,      #  (N, n=1, STATE_SIZE)
    #     hash_vec     #  (STATE_SIZE)
    # ) # (N, n) == [HASH(S1), HASH(S1), ..., HASH(SN), HASH(SN)]

    hashes = torch.mul(
        states,      #  (N, n=1, STATE_SIZE)
        hash_vec.unsqueeze(
            dim=0
        ).unsqueeze(
            dim=0
        ).expand(
            N, 
            1,
            state_size
        )
    ).sum(dim=2)  

    i = 0
    actions = torch.full(
        size=[N, 1], 
        fill_value=-1,
        dtype=torch.int64,         
        device=device
    )

    while i < n:
        action = torch.randint(low=0, high=n_gens, size=(N,), device=device)
        has_duplicates = True
        
        # new_states: torch.Tensor = torch.tensor(())
        new_states = states[:, -1, :]
        new_hashes = hashes[:, -1]

        while has_duplicates:
            new_states = torch.gather(
                input=states[:, -1, :],
                dim=1,
                index=generators[action]
            )
            
            # new_hashes = torch.einsum('NS,S->N', new_states, hash_vec)

            new_hashes = torch.mul(
                new_states, # [N, STATE_SIZE]
                hash_vec.unsqueeze(
                    dim=0
                ).expand(
                    N,
                    state_size
                )
            ).sum(dim=1) # [N]

            check_hashes = torch.eq(
                hashes, 
                new_hashes.unsqueeze(dim=1).expand(N, hashes.shape[1])
            )
            
            mask_duplicated_actions = check_hashes.any(dim=1)
            has_duplicates = mask_duplicated_actions.any().item() == True

            if has_duplicates:
                # print(f"{i}) Duplicated!")
                updated_actions = torch.randint(low=0, high=n_gens, size=(N,), device=device)
                action[mask_duplicated_actions] = updated_actions[mask_duplicated_actions]

                
            # if has_duplicates:
            #     print("Duplicated!")
            #     continue
            
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
        dtype=torch.float32,
        device=device
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

def scrambles_collate_fn(
    batch: torch.Tensor,
    n_gens: int = 3
):
    states = torch.cat([b[0] for b in batch], dim=0)
    actions = torch.cat([b[1] for b in batch], dim=0)
    lengths = torch.cat([b[2] for b in batch], dim=0)
    
    state_size = states.shape[-1]

    states = states.view(-1, state_size)
    actions = actions.view(-1)
    lengths = lengths.view(-1, 1)

    lengths = reverse_actions(lengths, n_gens=n_gens)

    return states, actions, lengths

class Cube3Dataset2(torch.utils.data.Dataset):
    def __init__(
            self, 
            n: int,
            N: int,
            size: int,
            generators: torch.Tensor
        ):
        
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

class Cube3Dataset3(torch.utils.data.Dataset):
    def __init__(
        self, 
        n: int,
        N: int,
        size: int,
        generators: torch.Tensor,
        device: torch.device
    ):
        self.n = n
        self.N = N
        self.state_size = generators.shape[1]
        self.generators = generators
        self.device=device
        self.hash_vec = torch.randint(
            0, 
            1_000_000_000_000, 
            (self.state_size,), 
            device=self.device
        )  

        self.size = size

    @torch.jit.export
    def __getitem__(self, idx):
        
        return get_torch_scrambles_3(
            N=self.N,
            n=self.n,
            generators=self.generators,
            hash_vec=self.hash_vec,
            device=self.device
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