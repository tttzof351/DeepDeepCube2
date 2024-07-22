import time

import torch
import torch.nn.functional as F
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
        alpha: float,
        generators: torch.Tensor,
        goal_state: torch.Tensor,
        device: torch.device
    ):
        self.model = model
        self.num_steps = num_steps
        self.beam_width = beam_width
        self.alpha = alpha
        self.generators = generators
        self.n_gens = generators.size(0)
        self.state_size = generators.size(1)
        self.device = device
        self.hash_vec = torch.randint(0, 1_000_000_000_000, (self.state_size,))
        self.goal_state = goal_state

        self.model.eval()
        self.model.to(device)

    def get_unique_states_idx(
        self, 
        states: torch.Tensor
    ) -> torch.Tensor:
        hashed = torch.sum(self.hash_vec * states, dim=1)
        hashed_sorted, idx = torch.sort(hashed)
        mask = torch.cat((torch.tensor([True]), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
        
        # count_removed = (mask == False).int().sum()
        # print(f"{self.global_i}) count_removed:", count_removed)

        return idx[mask]

    def expand_canditate_solutions(
        self, 
        candidate_solutions: torch.Tensor
    ):
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

        # print("states.size(0):", states.size(0))
        # print("candidate_solutions.size(1):", candidate_solutions.size(1))
        applied_actions = torch.arange(0, self.n_gens, dtype=torch.int64).expand(
            candidate_solutions.size(1), 
            self.n_gens
        )
        applied_actions = applied_actions.reshape(1, applied_actions.shape[0] * applied_actions.shape[1])
        candidate_solutions = torch.cat([expand_candidate_solutions, applied_actions]) 
        
        return candidate_solutions
    
    def expand_log_values(
        self, 
        log_values: torch.Tensor
    ):
        expanded_log_values = log_values.unsqueeze(dim=1).expand(
            log_values.shape[0],
            self.n_gens    
        ).reshape(
            log_values.shape[0] * self.n_gens
        )
        
        return expanded_log_values

    def get_neighbors(
        self, 
        states: torch.Tensor
    ) -> torch.Tensor:
        expanded_states = states.unsqueeze(1).expand(states.size(0), self.n_gens, self.state_size)        
        indexes = self.generators.unsqueeze(0).expand(states.size(0), self.n_gens, self.state_size)        
        
        new_states = torch.gather(input=expanded_states, dim=2, index=indexes)

        return new_states

    def batch_predict(
        self, 
        model: torch.nn.Module, 
        data: torch.Tensor, 
        device: torch.device,
        batch_size: int
    ) -> torch.Tensor:        
        n_samples = data.shape[0]
        values = []
        policies = []

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch = data[start:end].to(device)

            with torch.no_grad():
                batch_values, batch_policy = model(batch)
                batch_values = batch_values.squeeze(dim=1)

            values.append(batch_values)
            policies.append(batch_policy)

        values = torch.cat(values, dim=0)
        policies = torch.cat(policies, dim=0)
        return values, policies

    def predict_values(
        self, 
        states: torch.Tensor
    ) -> torch.Tensor:        
        values, policy = self.batch_predict(self.model, states, self.device, 4096)
        self.processed_states_count += states.shape[0]
        if (self.processed_states_count - self.printed_count > 1_000_000):
            count_millions = np.round(self.processed_states_count / 10**6, 3)
            print(f"{self.global_i}) Processed: {count_millions}M")
            self.printed_count = self.processed_states_count


        values = values.cpu()
        policy = policy.cpu()

        return values, policy
    
    def update_greedy_step(self) -> torch.Tensor:
        neighbors = self.get_neighbors(self.states)
        
        candidate_solutions = self.expand_canditate_solutions(self.candidate_solutions)        
        neighbors = neighbors.flatten(end_dim=1)

        # print(f"{self.global_i}) neighbours_states:", neighbors.shape)

        expanded_log_values = self.expand_log_values(self.parent_log_values)
        
        idx_uniq = self.get_unique_states_idx(neighbors)

        candidate_solutions = candidate_solutions[:, idx_uniq]
        neighbors = neighbors[idx_uniq]
        expanded_log_values = expanded_log_values[idx_uniq]
        
        pred_values, pred_policy = self.predict_values(neighbors)
        log_values = torch.log(pred_values)

        # print("log_values:", log_values.shape)
        # print("parent_log_values:", expanded_log_values.shape)
        
        log_scores = log_values + expanded_log_values * self.alpha
        idx = torch.argsort(log_scores)[:self.beam_width]

        candidate_solutions = candidate_solutions[:, idx]
        neighbors = neighbors[idx]
        log_scores = log_scores[idx]

        self.states = neighbors
        self.candidate_solutions = candidate_solutions
        self.parent_log_values = log_scores

        if self.global_i > 10:
            pass
            # exit()
        self.global_i += 1

    def search(
        self,
        state: torch.Tensor
    ):
        if len(state.shape) < 2:
            state = state.unsqueeze(0)

        self.candidate_solutions = torch.full(
            size=[1, 1], # (path of action, different solutions)
            fill_value=-1,
            dtype=torch.int64
        )
        self.processed_states_count = 0
        self.printed_count = 0
        self.parent_log_values = torch.tensor([0])
        self.global_i = 0

        self.model.eval()
        self.states = state.clone()
        for j in range(self.num_steps):
            self.update_greedy_step()
            search_result = (self.states == self.goal_state).all(dim=1).nonzero(as_tuple=True)[0]
            if (len(search_result) > 0):
                solution_index = search_result.item()
                solution = self.candidate_solutions[:, solution_index]
                return solution[1:], self.processed_states_count
        return None, self.processed_states_count

if __name__ == "__main__":
    deepcube_test = open_pickle("./assets/data/deepcubea/data_0.pkl")
    i = 0
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
        beam_width=100_000,
        alpha=0.0,
        goal_state=goal_state,
        device = "mps"
    )
    solution, processed_states_count = beam_search.search(state=state)
    count_millions = np.round(processed_states_count / 10**6, 3)
    end = time.time()
    duration = np.round(end - start, 3)

    print("solution_len:", len(solution))
    print(f"processed_states_count: {count_millions}M")
    print(f"duration: {duration} sec")
    
    print("state", state.shape)    
    for a in solution:
        state = state[:, generators[a]]
    
    print("solution:", solution)
    print("result_state:", state)
    print("state == goal_state:", (state == goal_state).all(dim=1))