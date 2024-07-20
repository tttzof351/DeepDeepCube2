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
        states: torch.Tensor,
        candidate_solutions: torch.Tensor
    ) -> torch.Tensor:
        hashed = torch.sum(self.hash_vec * states, dim=1)
        hashed_sorted, idx = torch.sort(hashed)
        mask = torch.cat((torch.tensor([True]), hashed_sorted[1:] - hashed_sorted[:-1] > 0))

        candidate_solutions = candidate_solutions[:, idx[mask]]
        return states[idx[mask]], candidate_solutions

    def get_neighbors(
        self, 
        states: torch.Tensor,
        candidate_solutions: torch.Tensor
    ) -> torch.Tensor:
        expanded_states = states.unsqueeze(1).expand(states.size(0), self.n_gens, self.state_size)
        
        expand_candidate_solutions = candidate_solutions.unsqueeze(2) # (path_of_action, different_solutions, 1)
        
        expand_candidate_solutions = expand_candidate_solutions.expand(
            candidate_solutions.size(0), 
            candidate_solutions.size(1),
            self.n_gens
        ) # (path of action, different solutions, ngens=12)
        
        expand_candidate_solutions = expand_candidate_solutions.reshape(
            candidate_solutions.size(0), 
            candidate_solutions.size(1) * self.n_gens
        ) # (path_of_action, different_solutions * ngens=12)

        indexes = self.generators.unsqueeze(0).expand(states.size(0), self.n_gens, self.state_size)        
        new_states = torch.gather(input=expanded_states, dim=2, index=indexes)

        applied_actions = torch.arange(0, self.n_gens, dtype=torch.int64).expand(states.size(0), self.n_gens)
        applied_actions = applied_actions.reshape(1, applied_actions.shape[0] * applied_actions.shape[1])
        candidate_solutions = torch.cat([expand_candidate_solutions, applied_actions]) 

        return new_states, candidate_solutions

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
        candidate_solutions: torch.Tensor, 
        beam_width: int = 1000
    ) -> torch.Tensor:
        neighbors, candidate_solutions = self.get_neighbors(states, candidate_solutions)
        
        neighbors = neighbors.flatten(end_dim=1)
        
        neighbors, candidate_solutions = self.get_unique_states(neighbors, candidate_solutions)
        
        y_pred = self.predict_clipped_values(neighbors)
        idx = torch.argsort(y_pred)[:beam_width]

        candidate_solutions = candidate_solutions[:, idx]
        neighbors = neighbors[idx]

        return neighbors, candidate_solutions

    def search(
        self,
        state: torch.Tensor
    ):
        if len(state.shape) < 2:
            state = state.unsqueeze(0)

        candidate_solutions = torch.full(
            size=[1, 1], # (path of action, different solutions)
            fill_value=-1,
            dtype=torch.int64
        )

        self.model.eval()
        states = state.clone()
        for j in range(self.num_steps):
            states, candidate_solutions = self.do_greedy_step(
                states=states, 
                candidate_solutions=candidate_solutions,
                beam_width=self.beam_width,
            )
            search_result = (states == self.goal_state).all(dim=1).nonzero(as_tuple=True)[0]
            if (len(search_result) > 0):
                solution_index = search_result.item()
                solution = candidate_solutions[:, solution_index]
                return solution[1:]
        return None

if __name__ == "__main__":
    deepcube_test = open_pickle("./assets/data/deepcubea/data_0.pkl")
    i = 1
    state = torch.tensor(deepcube_test['states'][i], dtype=torch.int64).unsqueeze(0)
    solution = deepcube_test['solutions'][i]    

    print("state:", state.shape)
    print("solution_len (optimal):", len(solution))

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
    solution = beam_search.search(state=state)
    end = time.time()
    duration = np.round(end - start, 3)

    print("solution_len:", len(solution))
    print("duration:", duration)
    
    print("state", state.shape)    
    for a in solution:
        state = state[:, generators[a]]
    
    print("solution:", solution)
    print("result_state:", state)
    print("state == goal_state:", (state == goal_state).all(dim=1))