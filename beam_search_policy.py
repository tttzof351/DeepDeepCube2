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

    def get_hashes(self, states: torch.Tensor):
        return torch.sum(self.hash_vec * states, dim=1)
    
    def get_unique_states_idx(
        self, 
        states: torch.Tensor
    ) -> torch.Tensor:
        hashed = torch.sum(self.hash_vec * states, dim=1)
        hashed_sorted, idx = torch.sort(hashed)
        mask = torch.cat((torch.tensor([True]), hashed_sorted[1:] - hashed_sorted[:-1] > 0))

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

    def predict(
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
        policy = torch.softmax(policy, dim=1).cpu()

        return values, policy
    
    def update_greedy_step(self):
        ######## ######## ######## ######## ########

        expanded_states = self.states.unsqueeze(dim=1).expand(
            self.states.shape[0],
            self.n_gens,
            self.states.shape[1],
        ).reshape(
            self.n_gens * self.states.shape[0],
            self.states.shape[1],
        ) # (N_STATES * N_GENS, STATE_SIZE) == [S1, S1, ..., SN, SN]

        expanded_actions = torch.arange(0, self.n_gens).unsqueeze(dim=0).expand(
            self.states.shape[0],
            self.n_gens
        ).reshape(
            self.states.shape[0] * self.n_gens
        ) # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
                
        neighbours_states = torch.gather(
            input=expanded_states,
            dim=1,
            index=self.generators[expanded_actions, :]
        ) # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]

        neighbors_policy_flatten = self.neighbors_policy.reshape(
            self.neighbors_policy.shape[0] * self.neighbors_policy.shape[1]
        ) # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]

        expanded_parent_cumulative_policy = self.parent_cumulative_policy.unsqueeze(dim=1).expand(
            self.parent_cumulative_policy.shape[0],
            self.n_gens
        ).reshape(
            self.n_gens * self.parent_cumulative_policy.shape[0]
        ) # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]

        scores = torch.log(neighbors_policy_flatten) + expanded_parent_cumulative_policy * 0.5 # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]

        neighbours_hashes = self.get_hashes(neighbours_states)
        unique_hahes_mask = [h not in self.processed_states for h in neighbours_hashes.tolist()]
        unique_hahes_mask = torch.tensor(unique_hahes_mask)
        
        neighbours_hashes = neighbours_hashes[unique_hahes_mask]
        expanded_actions = expanded_actions[unique_hahes_mask] # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
        neighbours_states = neighbours_states[unique_hahes_mask, :] # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]
        neighbors_policy_flatten = neighbors_policy_flatten[unique_hahes_mask] # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]
        expanded_parent_cumulative_policy = expanded_parent_cumulative_policy[unique_hahes_mask] # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
        scores = scores[unique_hahes_mask] # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]

        hashed_sorted, idx = torch.sort(neighbours_hashes)
        unique_hahes_mask_2 = torch.cat((torch.tensor([True]), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
        neighbours_hashes = neighbours_hashes[unique_hahes_mask_2]
        expanded_actions = expanded_actions[unique_hahes_mask_2] # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
        neighbours_states = neighbours_states[unique_hahes_mask_2, :] # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]
        neighbors_policy_flatten = neighbors_policy_flatten[unique_hahes_mask_2] # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]
        expanded_parent_cumulative_policy = expanded_parent_cumulative_policy[unique_hahes_mask_2] # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
        scores = scores[unique_hahes_mask_2] # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]


        # scores_idx = torch.argsort(scores, descending=True)#[:100_000] # beam width TODO
        
        # expanded_actions = expanded_actions[scores_idx] # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
        # neighbours_states = neighbours_states[scores_idx, :] # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]
        # neighbors_policy_flatten = neighbors_policy_flatten[scores_idx] # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]
        # expanded_parent_cumulative_policy = expanded_parent_cumulative_policy[scores_idx] # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
        # scores = scores[scores_idx] # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]
        # neighbours_hashes = neighbours_hashes[scores_idx]
        
        v, p = self.predict(neighbours_states) # (N_STATES)    
        
        v_idx = torch.argsort(v, descending=False)[:100_000]
        
        expanded_actions = expanded_actions[v_idx] # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
        neighbours_states = neighbours_states[v_idx, :] # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]
        neighbors_policy_flatten = neighbors_policy_flatten[v_idx] # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]
        expanded_parent_cumulative_policy = expanded_parent_cumulative_policy[v_idx] # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
        scores = scores[v_idx] # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]
        neighbours_hashes = neighbours_hashes[v_idx]
        v = v[v_idx]
        p = p[v_idx, :]
        
        self.value = v # (N_STATES) - one value for one state
        self.states = neighbours_states  # (N_STATES, STATE_SIZE)
        self.neighbors_policy = p # (N_STATES, N_GENS) - ~12 floats for one state
        self.parent_cumulative_policy = scores

        # print(f"{self.global_i}) v:", v[:3])
        # print(f"{self.global_i}) s:", scores[:3])

        for h in neighbours_hashes.tolist():
            # self.processed_states.add(h)
            pass

        ######## ######## ######## ######## ########        

        if self.global_i > 5:
            pass
            # exit()
        self.global_i += 1


    def print_shapes(self):
        print(f"{self.global_i}) value:", self.value.shape)
        print(f"{self.global_i}) neighbors_policy:", self.neighbors_policy.shape)
        print(f"{self.global_i}) solutions:", self.solutions.shape)
        print(f"{self.global_i}) solution_lengths:", self.solution_lengths.shape)
        print(f"{self.global_i}) parent_cumulative_value:", self.parent_cumulative_value.shape)
        print(f"{self.global_i}) parent_cumulative_policy:", self.parent_cumulative_policy.shape)
        print(f"{self.global_i}) processed_states:", len(self.processed_states))

    def search(
        self,
        state: torch.Tensor
    ):
        if len(state.shape) < 2:
            state = state.unsqueeze(0)

        self.model.eval()
        self.states = state.clone() # (N_STATES, STATE_SIZE)
        
        ########################################################
        
        self.processed_states_count = 0
        self.printed_count = 0
        
        ########################################################

        self.policy = torch.zeros((1)) # (N_STATES) - one probability for one state
        
        v, p = self.predict(self.states)  
        self.value = v#torch.zeros((1)) # (N_STATES) - one value for one state
        self.neighbors_policy = p#torch.zeros((1, self.n_gens)) # (N_STATES, N_GENS) - ~12 floats for one state
        
        self.solutions = torch.full(
            size=[1, 1], # (different solutions, path of action)
            fill_value=-1,
            dtype=torch.int64
        ) # (N_STATES, N_LENGTH) - one path for each state
        self.solution_lengths = torch.zeros((1)) # (N_STATES) - one float for one state
        self.parent_cumulative_value = torch.zeros((1)) # (N_STATES) - one float for one state
        self.parent_cumulative_policy = torch.zeros((1)) # (N_STATES) - one float for one state
        hashes = self.get_hashes(self.states).tolist()
        self.processed_states = set()
        for h in hashes:
            self.processed_states.add(h)


        self.global_i = 0

        for j in range(self.num_steps):
            self.update_greedy_step()
            search_result = (self.states == self.goal_state).all(dim=1).nonzero(as_tuple=True)[0]
            if (len(search_result) > 0):
                print("Found! j:", j)
                solution_index = search_result.item()
                solution = self.candidate_solutions[:, solution_index]
                return solution[1:], self.processed_states_count
            
        print("Not found!")
        return None, self.processed_states_count

if __name__ == "__main__":
    deepcube_test = open_pickle("./assets/data/deepcubea/data_0.pkl")
    i = 42
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
        num_steps=1000,
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