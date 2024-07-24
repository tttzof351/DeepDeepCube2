import time

import torch
import torch.nn.functional as F
import numpy as np

from utils import open_pickle

from cube3_game import Cube3Game
from models import Pilgrim
from datasets import get_torch_scrambles
from utils import set_seed


class BeamSearchMix:
    def __init__(
        self,
        model: torch.nn.Module,
        num_steps: int, 
        value_beam_width: int,
        policy_beam_width: int,
        alpha: float,
        generators: torch.Tensor,
        goal_state: torch.Tensor,
        verbose: bool,
        device: torch.device
    ):
        self.model = model
        self.num_steps = num_steps
        self.value_beam_width = value_beam_width
        self.policy_beam_width = policy_beam_width
        self.alpha = alpha
        self.generators = generators
        self.n_gens = generators.size(0)
        self.state_size = generators.size(1)
        self.device = device
        self.hash_vec = torch.randint(0, 1_000_000_000_000, (self.state_size,))
        self.goal_state = goal_state
        self.verbose = verbose

        self.model.eval()

    def get_hashes(self, states: torch.Tensor):
        return torch.sum(self.hash_vec * states, dim=1)
        
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
        self.processed_count += states.shape[0]
        if (self.processed_count - self.printed_count > 1_000_000):
            count_millions = np.round(self.processed_count / 10**6, 3)
            if self.verbose:
                print(f"{self.global_i}) Processed: {count_millions}M")
            self.printed_count = self.processed_count

        values = values.cpu()
        policy = torch.softmax(policy, dim=1).cpu()

        return values, policy
    
    def update_greedy_step(self):
        neighbours_states = self.states.unsqueeze(dim=1).expand(
            self.states.shape[0],
            self.n_gens,
            self.states.shape[1],
        ).reshape(
            self.n_gens * self.states.shape[0],
            self.states.shape[1],
        ) # (N_STATES * N_GENS, STATE_SIZE) == [S1, S1, ..., SN, SN]

        expanded_solution = self.solutions.unsqueeze(dim=1).expand(
            self.solutions.shape[0],
            self.n_gens, 
            self.solutions.shape[1]
        ).reshape(
            self.n_gens * self.solutions.shape[0],
            self.solutions.shape[1]
        ) # (N_STATES * N_GENS, SOLUTION_LEN) == [SOLUTION(S1), SOLUTION(S1), ..., SOLUTION(SN), SOLUTION(SN)]

        expanded_actions = torch.arange(0, self.n_gens).unsqueeze(dim=0).expand(
            self.states.shape[0],
            self.n_gens
        ).reshape(
            self.states.shape[0] * self.n_gens
        ) # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
                
        neighbours_states = torch.gather(
            input=neighbours_states,
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
        
        ###### ###### ###### ######

        # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]
        policy_scores = torch.log(neighbors_policy_flatten) + expanded_parent_cumulative_policy * 1.0 

        neighbours_hashes = self.get_hashes(neighbours_states)
        unique_hahes_mask = [h not in self.processed_hashes for h in neighbours_hashes.tolist()]
        for h in neighbours_hashes.tolist():
            self.processed_hashes.add(h)
            # pass

        unique_hahes_mask = torch.tensor(unique_hahes_mask)
        
        neighbours_hashes = neighbours_hashes[unique_hahes_mask]
        expanded_actions = expanded_actions[unique_hahes_mask] # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
        expanded_solution = expanded_solution[unique_hahes_mask] # (N_STATES * N_GENS, SOLUTION_LEN) == [SOLUTION(S1), SOLUTION(S1), ..., SOLUTION(SN), SOLUTION(SN)]
        neighbours_states = neighbours_states[unique_hahes_mask, :] # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]
        neighbors_policy_flatten = neighbors_policy_flatten[unique_hahes_mask] # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]
        expanded_parent_cumulative_policy = expanded_parent_cumulative_policy[unique_hahes_mask] # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
        policy_scores = policy_scores[unique_hahes_mask] # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]

        hashed_sorted, hashed_idx = torch.sort(neighbours_hashes)
        unique_hahes_mask_2 = torch.cat((torch.tensor([True]), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
        hashed_idx = hashed_idx[unique_hahes_mask_2]
        
        neighbours_hashes = neighbours_hashes[hashed_idx] # (N_GENS * N_STATES)
        expanded_actions = expanded_actions[hashed_idx] # (N_GENS * N_STATES) == [A1, A2, ..., A1, A2]
        expanded_solution = expanded_solution[hashed_idx] # (N_STATES * N_GENS, SOLUTION_LEN) == [SOLUTION(S1), SOLUTION(S1), ..., SOLUTION(SN), SOLUTION(SN)]
        neighbours_states = neighbours_states[hashed_idx, :] # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]
        neighbors_policy_flatten = neighbors_policy_flatten[hashed_idx] # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]
        expanded_parent_cumulative_policy = expanded_parent_cumulative_policy[hashed_idx] # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
        policy_scores = policy_scores[hashed_idx] # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]

        if self.policy_beam_width is not None:
            policy_scores_idx = torch.argsort(policy_scores, descending=True)[:self.policy_beam_width] # beam width TODO

            expanded_actions = expanded_actions[policy_scores_idx] # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
            expanded_solution = expanded_solution[policy_scores_idx] # (N_STATES * N_GENS, SOLUTION_LEN) == [SOLUTION(S1), SOLUTION(S1), ..., SOLUTION(SN), SOLUTION(SN)]
            neighbours_states = neighbours_states[policy_scores_idx, :] # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]
            neighbors_policy_flatten = neighbors_policy_flatten[policy_scores_idx] # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]
            expanded_parent_cumulative_policy = expanded_parent_cumulative_policy[policy_scores_idx] # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
            policy_scores = policy_scores[policy_scores_idx] # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]
            neighbours_hashes = neighbours_hashes[policy_scores_idx]
        
        v, p = self.predict(neighbours_states) # (N_STATES)    
        
        if self.value_beam_width is not None:
            v_idx = torch.argsort(v)[:self.value_beam_width]

            expanded_actions = expanded_actions[v_idx] # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
            expanded_solution = expanded_solution[v_idx] # (N_STATES * N_GENS, SOLUTION_LEN) == [SOLUTION(S1), SOLUTION(S1), ..., SOLUTION(SN), SOLUTION(SN)]            
            neighbours_states = neighbours_states[v_idx, :] # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]
            neighbors_policy_flatten = neighbors_policy_flatten[v_idx] # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]
            expanded_parent_cumulative_policy = expanded_parent_cumulative_policy[v_idx] # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
            policy_scores = policy_scores[v_idx] # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]
            neighbours_hashes = neighbours_hashes[v_idx]
            v = v[v_idx]
            p = p[v_idx, :]
        
        self.value = v # (N_STATES) - one value for one state
        self.states = neighbours_states  # (N_STATES, STATE_SIZE)
        self.neighbors_policy = p # (N_STATES, N_GENS) - ~12 floats for one state
        
        self.solutions = torch.cat([expanded_solution, expanded_actions.unsqueeze(dim=1)], dim=1)
        self.parent_cumulative_policy = policy_scores

        if self.verbose:
            print(f"{self.global_i}) v:", v[:3], "; policy_s:", policy_scores[:3] )        

        ######## ######## ######## ######## ########        

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
        
        self.processed_count = 0
        self.printed_count = 0
        
        ########################################################

        # self.policy = torch.zeros((1)) # (N_STATES) - one probability for one state
        
        v, p = self.predict(self.states)  
        
        self.value = v # (N_STATES) - one value for one state
        self.neighbors_policy = p # (N_STATES, N_GENS) - ~12 floats for one state
        
        self.solutions = torch.full(
            size=[1, 1], # (different solutions, path of action)
            fill_value=-1,
            dtype=torch.int64
        ) # (N_STATES, N_LENGTH) - one path for each state

        # self.solution_lengths = torch.zeros((1)) # (N_STATES) - one float for one state
        self.parent_cumulative_value = torch.zeros((1)) # (N_STATES) - one float for one state
        self.parent_cumulative_policy = torch.zeros((1)) # (N_STATES) - one float for one state

        ########################################################
        
        hashes = self.get_hashes(self.states).tolist()
        self.processed_hashes = set()
        for h in hashes:
            self.processed_hashes.add(h)

        self.global_i = 0

        for j in range(self.num_steps):
            self.update_greedy_step()
            search_result = (self.states == self.goal_state).all(dim=1).nonzero(as_tuple=True)[0]
            if (len(search_result) > 0):                
                solution_index = search_result.item()
                solution = self.solutions[solution_index, :]
                return solution[1:], self.processed_count
            
        return None, self.processed_count

if __name__ == "__main__":
    set_seed(0)
    deepcube_test = open_pickle("./assets/data/deepcubea/data_0.pkl")
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    generators = torch.tensor(game.actions, dtype=torch.int64)

    i = 1
    state = torch.tensor(deepcube_test['states'][i], dtype=torch.int64).unsqueeze(0)
    solution = deepcube_test['solutions'][i]    

    print("state:", state.shape)
    print("solution_len (optimal):", len(solution))

    # states, actions, values = get_torch_scrambles(
    #     n = 1,
    #     space_size = game.space_size,
    #     action_size = game.action_size,
    #     length = 20,
    #     permutations = generators
    # )
    # state = states[-1, :]
    # value = values[-1].item()
    # print("value:", value)

    device = "mps"
    model = Pilgrim()
    model.to(device)
    # model.load_state_dict(torch.load("./assets/models/Cube3ResnetModel.pt"))

    mode = "value"
    models = {
        "value": "./assets/models/Cube3ResnetModel_value.pt",
        "policy": "./assets/models/Cube3ResnetModel_policy.pt"
    }

    model.load_state_dict(torch.load(models[mode]))

    goal_state = torch.arange(0, 54, dtype=torch.int64)
    
    start = time.time()

    beam_search = BeamSearchMix(
        model=model,
        generators=generators,
        num_steps=100_000_000,
        value_beam_width=100_000 if mode == "value" else None,
        policy_beam_width=100_000 if mode == "policy" else None,
        alpha=0.0,
        goal_state=goal_state,
        verbose=True,
        device=device
    )
    solution, processed_count = beam_search.search(state=state)

    end = time.time()

    duration = np.round(end - start, 3)

    print("solution_len:", len(solution))
    count_millions = np.round(processed_count / 10**6, 3)
    print(f"processed_count: {count_millions}M")
    print(f"duration: {duration} sec")
    
    s = state.squeeze(0).clone()
    for a in solution.tolist():
        s = s[generators[a]]
    print("out state is goal?:", (s == goal_state).all().item())
