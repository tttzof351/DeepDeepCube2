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
        gen_indexes: torch.Tensor
    ) -> torch.Tensor:
        hashed = torch.sum(self.hash_vec * states, dim=1)
        hashed_sorted, idx = torch.sort(hashed)
        mask = torch.cat((torch.tensor([True]), hashed_sorted[1:] - hashed_sorted[:-1] > 0))

        gen_indexes = gen_indexes[:, idx[mask]]
        return states[idx][mask], gen_indexes

    def get_neighbors(
        self, 
        states: torch.Tensor,
        parent_actions: torch.Tensor
    ) -> torch.Tensor:
        expanded_states = states.unsqueeze(1).expand(states.size(0), self.n_gens, self.state_size)
        expand_parent_actions = parent_actions.unsqueeze(2).expand(
            parent_actions.size(0), 
            parent_actions.size(1),
            self.n_gens
        )
        # print("before expand_parent_actions:", expand_parent_actions.shape)
        expand_parent_actions = expand_parent_actions.reshape(parent_actions.size(0), parent_actions.size(1) * self.n_gens)
        # print("after expand_parent_actions:", expand_parent_actions.shape)

        indexes = self.generators.unsqueeze(0).expand(states.size(0), self.n_gens, self.state_size)        
        gen_indexes = torch.arange(0, self.n_gens, dtype=torch.int64).expand(states.size(0), self.n_gens)
        new_states = torch.gather(input=expanded_states, dim=2, index=indexes)

        gen_indexes = gen_indexes.reshape(1, gen_indexes.shape[0] * gen_indexes.shape[1])
        # print("gen_indexes before cat:", gen_indexes.shape)
        gen_indexes = torch.cat([expand_parent_actions, gen_indexes])
        # print("gen_indexes after:", gen_indexes.shape)

        # exit()

        return new_states, gen_indexes

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
        # print("parent_actions:", self.parent_actions.shape)
        neighbors, gen_indexes = self.get_neighbors(states, self.parent_actions)
        
        neighbors = neighbors.flatten(end_dim=1)
        # gen_indexes = gen_indexes.flatten(end_dim=1)
        
        neighbors, gen_indexes = self.get_unique_states(neighbors, gen_indexes)
        # print("final gen_indexes:", gen_indexes.shape)
        
        y_pred = self.predict_clipped_values(neighbors)
        idx = torch.argsort(y_pred)[:B]

        self.parent_actions = gen_indexes[:, idx[:B]]

        return neighbors[idx]

    def search(
        self,
        state: torch.Tensor
    ):
        if len(state.shape) < 2:
            state = state.unsqueeze(0)

        self.parent_actions = torch.full(
            size=[1, 1],
            fill_value=-1,
            dtype=torch.int64
        )

        self.model.eval()
        states = state.clone()
        for j in range(self.num_steps):
            states = self.do_greedy_step(states, self.beam_width)
            has_goal = (states == self.goal_state).all(dim=1)
            search_result = has_goal.nonzero(as_tuple=True)[0]
            # print("has goal:", tuple_goal)
            # if (states == self.goal_state).all(dim=1).any():
            if (len(search_result) > 0):
                # print("len(search_result)", len(search_result))
                solution_index = search_result.item()
                solution = self.parent_actions[:, solution_index]
                print("solution_index:", solution_index)
                print("solution:", solution)                
                return solution[1:]
        return None

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
    solution = beam_search.search(state=state)
    end = time.time()
    duration = np.round(end - start, 3)

    print("found_len:", solution)
    print("duration:", duration)
    
    print("state", state.shape)    
    for a in solution:
        state = state[:, generators[a]]
    
    print(solution)
    print(state)
    print("state == goal_state:", (state == goal_state).all(dim=1))