import time

import torch
import numpy as np

from utils import open_pickle

from cube3_game import Cube3Game
from models import Pilgrim


class BeamSearch:
    def __init__(
        self,
        model: torch.nn.Module,
        num_steps: int, 
        beam_width: int,
        generators: torch.Tensor,
        goal_state: torch.Tensor,
        device: torch.device
    ):
        self.model = model
        self.num_steps = num_steps
        self.beam_width = beam_width
        self.generators = generators
        self.n_gens = generators.size(0)
        self.state_size = generators.size(1)
        self.device = device
        self.hash_vec = torch.randint(0, 1_000_000_000_000, (self.state_size,))
        self.goal_state = goal_state

        self.model.eval()
        self.model.to(device)

    def get_unique_states(
        self, 
        states: torch.Tensor
    ) -> torch.Tensor:
        hashed = torch.sum(self.hash_vec * states, dim=1)
        hashed_sorted, idx = torch.sort(hashed)
        mask = torch.cat((torch.tensor([True]), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
        return states[idx][mask]

    def get_neighbors(
        self, 
        states: torch.Tensor
    ) -> torch.Tensor:
        return torch.gather(
            states.unsqueeze(1).expand(states.size(0), self.n_gens, self.state_size), 
            2, 
            self.generators.unsqueeze(0).expand(states.size(0), self.n_gens, self.state_size)
        )

    def batch_predict(
        self, 
        model: torch.nn.Module, 
        data: torch.Tensor, 
        device: torch.device,
        batch_size: int
    ) -> torch.Tensor:        
        n_samples = data.shape[0]
        outputs = []

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch = data[start:end].to(device)

            with torch.no_grad():
                batch_output, _  = model(batch)
                batch_output = batch_output.flatten()

            outputs.append(batch_output)

        final_output = torch.cat(outputs, dim=0)
        return final_output

    def predict_values(
        self, 
        states: torch.Tensor
    ) -> torch.Tensor:
        return self.batch_predict(self.model, states, self.device, 4096).cpu()

    def predict_clipped_values(
        self, 
        states: torch.Tensor
    ) -> torch.Tensor:
        return torch.clip(self.predict_values(states), 0, torch.inf)

    def do_greedy_step(
        self, 
        states: torch.Tensor, 
        B: int = 1000
    ) -> torch.Tensor:
        neighbors = self.get_neighbors(states).flatten(end_dim=1)
        neighbors = self.get_unique_states(neighbors)
        y_pred = self.predict_clipped_values(neighbors)
        idx = torch.argsort(y_pred)[:B]
        return neighbors[idx]

    def search(
        self,
        state: torch.Tensor
    ):
        if len(state.shape) < 2:
            state = state.unsqueeze(0)

        self.model.eval()
        states = state.clone()
        for j in range(self.num_steps):
            states = self.do_greedy_step(states, self.beam_width)
            if (states == self.goal_state).all(dim=1).any():
                return j
        return -1

if __name__ == "__main__":
    deepcube_test = open_pickle("./assets/data/deepcubea/data_0.pkl")
    i = 0
    state = torch.tensor(deepcube_test['states'][i], dtype=torch.int64).unsqueeze(0)
    solution = deepcube_test['solutions'][i]    

    print("state:", state.shape)
    print("solution len:", len(solution))

    model = Pilgrim()
    model.load_state_dict(torch.load("./assets/models/Cube3ResnetModel.pt"))

    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    
    generators = torch.tensor(game.actions, dtype=torch.int64)
    goal_state = torch.arange(0, 54, dtype=torch.int64)
    
    start = time.time()
    beam_search = BeamSearch(
        model=model,
        generators=generators,
        num_steps=100,
        beam_width=100000,
        goal_state=goal_state,
        device = "mps"
    )
    found_len = beam_search.search(state=state)
    end = time.time()
    duration = np.round(end - start, 3)

    print("found_len:", found_len)
    print("duration:", duration)