import sys
sys.path.append("..")  # Add parent directory to system path


from utils import *
from cube3_game import Cube3Game
from models import Pilgrim, PilgrimTransformer, PilgrimSimple, PilgrimCNN, PilgrimMLP2

import numpy as np
import torch

from g_datasets import *
import importlib
import a_search_mix

# Reload the a_search_mix module
importlib.reload(a_search_mix)

# Now you can use AStarVector from the reloaded module
from a_search_mix import AStarVector

if __name__ == "__main__":
    optimum = open_pickle("../assets/data/deepcubea/data_0.pkl")
    deepcube_res = open_pickle("../assets/data/deepcubea/results.pkl")
    report = open_pickle("../assets/reports/Cube3ResnetModel_value_policy_3_8B_14M_search_value_full.pkl")
    game = Cube3Game("../assets/envs/qtm_cube3.pickle")
    generators = torch.tensor(game.actions, dtype=torch.int64)

    state_size = game.actions.shape[1]
    hash_vec = torch.randint(0, 1_000_000_000_000, (state_size,))    

    set_seed(42)

    scrambles, actions, lengths = get_torch_scrambles_3(
        N=10,
        n=30,
        generators=generators,
        hash_vec=hash_vec,
        device="cpu"
    )
    
    save_pickle({
        "scrambles": scrambles.detach().numpy(),
        "actions": actions.detach().numpy(),
        "lengths": lengths.detach().numpy()
    }, "../assets/lengths/scrambles_dict.pkl")

    model_device = "mps"
    model = Pilgrim(
        input_dim = 54, 
        hidden_dim1 = 5000, 
        hidden_dim2 = 1000, 
        num_residual_blocks = 4 
    ) # ~14M

    model.load_state_dict(
        torch.load(
            "../assets/models/Cube3ResnetModel_value_policy_3_8B_14M.pt",
            map_location=model_device)
    )
    model = model.to(model_device)
    goal_state = torch.arange(0, 54, dtype=torch.int64)

    solutions = []
    for i, scramble in tqdm(enumerate(scrambles)):
        path_finder = AStarVector(
            model=model,
            generators=generators,
            num_steps=10_000,
            b_exp=10_0000,
            b_keep=1_000,
            temperature=0.0,
            goal_state=goal_state,
            verbose=False,
            device="cpu",
            model_device="mps"
        )

        solution, processed_count = path_finder.search(scramble.unsqueeze(0))
        solutions.append(solution.detach().tolist())
    
    save_pickle({
        "path_finder_solutions": solutions,
    }, "../assets/lengths/path_finder_solutions.pkl")
    
