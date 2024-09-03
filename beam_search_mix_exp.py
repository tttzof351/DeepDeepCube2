import time

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from contextlib import nullcontext

from utils import open_pickle

from cube3_game import Cube3Game
from models import Pilgrim
from g_datasets import get_torch_scrambles_3
from utils import set_seed
from utils import save_pickle
from utils import TimeContext
from utils import int_to_human


class BeamSearchMixExp:
    def __init__(
        self,
        model: torch.nn.Module,
        num_steps: int, 
        value_beam_width: int,
        policy_beam_width: int,
        alpha: float,
        T: float,
        B: int,
        generators: torch.Tensor,
        goal_state: torch.Tensor,
        verbose: bool,
        model_device: torch.device,
        device: torch.device
    ):
        self.device = device
        self.model_device = model_device
        
        self.model = model
        self.use_amp = str(model_device) == "cuda"
        # print("self.use_amp:", self.use_amp)

        self.num_steps = num_steps
        self.value_beam_width = value_beam_width
        self.policy_beam_width = policy_beam_width
        self.alpha = alpha
        self.T = T  
        self.B = B
        self.generators = generators.to(self.device)
        self.n_gens = generators.size(0)
        self.state_size = generators.size(1)
        self.hash_vec = torch.randint(0, 1_000_000_000_000, (self.state_size,), device=self.device)
        self.goal_state = goal_state.to(self.device)
        self.verbose = verbose

        self.model.to(self.model_device)
        self.model.eval()

        if self.use_amp:
            self.ctx = torch.autocast(device_type=self.model_device, dtype=torch.float16, enabled=self.use_amp)
        else:
            self.ctx = nullcontext()        

    def get_hashes(self, states: torch.Tensor):
        return torch.sum(self.hash_vec * states, dim=1)
        
    def batch_predict(
        self, 
        model: torch.nn.Module, 
        data: torch.Tensor, 
        batch_size: int
    ) -> torch.Tensor:        
        n_samples = data.shape[0]
        values = []
        policies = []

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch = data[start:end].to(self.model_device)

            with self.ctx:
                with torch.no_grad():
                    batch_values, batch_policy = model(batch)
                    batch_values = batch_values.squeeze(dim=1)

                    values.append(batch_values)
                    policies.append(batch_policy)

        values = torch.cat(values, dim=0).to(self.device).detach()
        policies = torch.cat(policies, dim=0).to(self.device).detach()
        return values, policies

    def predict(
        self, 
        states: torch.Tensor
    ) -> torch.Tensor:
        values, policy = self.batch_predict(self.model, states, 4096)
        self.processed_count += states.shape[0]
        if (self.processed_count - self.printed_count > 1_000_000):
            count_millions = np.round(self.processed_count / 10**6, 3)
            # if self.verbose:
            #     print(f"{self.global_i}) Processed: {count_millions}M")
            self.printed_count = self.processed_count

        values = values.to(self.device)#.cpu()
        policy = torch.softmax(policy.float(), dim=1).to(self.device)#.cpu()

        return values, policy
    
    def update_greedy_step(self):
        with TimeContext(f"{self.global_i}) Expanded states", self.verbose):
            neighbours_states = self.states.unsqueeze(dim=1).expand(
                self.states.shape[0],
                self.n_gens,
                self.states.shape[1],
            ).reshape(
                self.n_gens * self.states.shape[0],
                self.states.shape[1],
            ).to(self.device) # (N_STATES * N_GENS, STATE_SIZE) == [S1, S1, ..., SN, SN]

            expanded_solution = self.solutions.unsqueeze(dim=1).expand(
                self.solutions.shape[0],
                self.n_gens, 
                self.solutions.shape[1]
            ).reshape(
                self.n_gens * self.solutions.shape[0],
                self.solutions.shape[1]
            ).to(self.device) # (N_STATES * N_GENS, SOLUTION_LEN) == [SOLUTION(S1), SOLUTION(S1), ..., SOLUTION(SN), SOLUTION(SN)]

            expanded_actions = torch.arange(0, self.n_gens).unsqueeze(dim=0).expand(
                self.states.shape[0],
                self.n_gens
            ).reshape(
                self.states.shape[0] * self.n_gens
            ).to(self.device) # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
                    
            neighbours_states = torch.gather(
                input=neighbours_states,
                dim=1,
                index=self.generators[expanded_actions, :]
            ).to(self.device) # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]

            neighbors_policy_flatten = self.neighbors_policy.reshape(
                self.neighbors_policy.shape[0] * self.neighbors_policy.shape[1]
            ).to(self.device) # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]

            expanded_parent_cumulative_policy = self.parent_cumulative_policy.unsqueeze(dim=1).expand(
                self.parent_cumulative_policy.shape[0],
                self.n_gens
            ).reshape(
                self.n_gens * self.parent_cumulative_policy.shape[0]
            ).to(self.device) # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
        
        ###### ###### ###### ######

        with TimeContext(f"{self.global_i}) Calculate policy score", self.verbose):
            # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]
            policy_scores = torch.log(neighbors_policy_flatten) + expanded_parent_cumulative_policy * self.alpha

        with TimeContext(f"{self.global_i}) Calculate hash", self.verbose):
            neighbours_hashes = self.get_hashes(neighbours_states)

        # with TimeContext(f"{self.global_i}) Global unique hash:", self.verbose):
        #     unique_hahes_mask = [h not in self.processed_hashes for h in neighbours_hashes.tolist()]
        #     for h in neighbours_hashes.tolist():
        #         self.processed_hashes.add(h)
        #         # pass

        #     unique_hahes_mask = torch.tensor(unique_hahes_mask, device=self.device)
            
        #     neighbours_hashes = neighbours_hashes[unique_hahes_mask]
        #     expanded_actions = expanded_actions[unique_hahes_mask] # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
        #     expanded_solution = expanded_solution[unique_hahes_mask] # (N_STATES * N_GENS, SOLUTION_LEN) == [SOLUTION(S1), SOLUTION(S1), ..., SOLUTION(SN), SOLUTION(SN)]
        #     neighbours_states = neighbours_states[unique_hahes_mask, :] # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]
        #     neighbors_policy_flatten = neighbors_policy_flatten[unique_hahes_mask] # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]
        #     expanded_parent_cumulative_policy = expanded_parent_cumulative_policy[unique_hahes_mask] # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
        #     policy_scores = policy_scores[unique_hahes_mask] # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]

        with TimeContext(f"{self.global_i}) Double states hash", self.verbose):
            hashed_sorted, hashed_idx = torch.sort(neighbours_hashes)
            unique_hahes_mask_2 = torch.cat((torch.tensor([True], device=self.device), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
            hashed_idx = hashed_idx[unique_hahes_mask_2]
            
            neighbours_hashes = neighbours_hashes[hashed_idx] # (N_GENS * N_STATES)
            expanded_actions = expanded_actions[hashed_idx] # (N_GENS * N_STATES) == [A1, A2, ..., A1, A2]
            expanded_solution = expanded_solution[hashed_idx] # (N_STATES * N_GENS, SOLUTION_LEN) == [SOLUTION(S1), SOLUTION(S1), ..., SOLUTION(SN), SOLUTION(SN)]
            neighbours_states = neighbours_states[hashed_idx, :] # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]
            neighbors_policy_flatten = neighbors_policy_flatten[hashed_idx] # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]
            expanded_parent_cumulative_policy = expanded_parent_cumulative_policy[hashed_idx] # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
            policy_scores = policy_scores[hashed_idx] # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]

        # if self.policy_beam_width is not None:
        #     with TimeContext(f"{self.global_i}) Policy filter", self.verbose):
        #         policy_scores_idx = torch.argsort(policy_scores, descending=True)#[:self.policy_beam_width] 

        #         expanded_actions = expanded_actions[policy_scores_idx] # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
        #         expanded_solution = expanded_solution[policy_scores_idx] # (N_STATES * N_GENS, SOLUTION_LEN) == [SOLUTION(S1), SOLUTION(S1), ..., SOLUTION(SN), SOLUTION(SN)]
        #         neighbours_states = neighbours_states[policy_scores_idx, :] # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]
        #         neighbors_policy_flatten = neighbors_policy_flatten[policy_scores_idx] # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]
        #         expanded_parent_cumulative_policy = expanded_parent_cumulative_policy[policy_scores_idx] # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
        #         policy_scores = policy_scores[policy_scores_idx] # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]
        #         # neighbours_hashes = neighbours_hashes[policy_scores_idx]
        
        with TimeContext(f"{self.global_i}) Inference", self.verbose):
            v, p = self.predict(neighbours_states) # (N_STATES)    
        
        if self.value_beam_width is not None:
            with TimeContext(f"{self.global_i}) Value+Policy filter", self.verbose):                
                # B = 2048

                # if self.global_i < 5:
                #     B = 20_000
                # elif self.global_i < 10:
                #     B = 100_000
                # elif self.global_i < 15:
                #     B = 200_000
                # else:
                #     B = 50_000
                
                # B = int(B / 10)
                # B = 4

                v_min = torch.min(v)
                policy_score_max = torch.max(policy_scores)
                # v - чем меньше, тем лучше
                # poilcy_score - чем больше тем лучше
                # V+P - чем меньше тем лучше
                
                # value_policy_score = v + (-policy_scores) * 0.0
                # value_policy_score = 2.0 * torch.softmax(v, dim=0) + torch.softmax(-policy_scores, dim=0)                 
                # v_policy_idx = torch.argsort(value_policy_score)[:B]
                
                # value_policy_score = torch.softmax(
                #     -v,
                #     dim=0
                # )
                # v_policy_idx = value_policy_score.multinomial(num_samples=B, replacement=True)#[:B]
                value_policy_score = torch.softmax(
                    #torch.softmax(-v, dim=0) + torch.softmax(+policy_scores, dim=0) * 0.0,
                    (-v * 0.0 + policy_scores * 1.0) / self.T,
                    # policy_scores,
                    # -v,
                    dim=0
                )
                v_policy_idx = value_policy_score.multinomial(num_samples=min(self.B, value_policy_score.shape[0]), replacement=True)#[:B]


                expanded_actions = expanded_actions[v_policy_idx] # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]
                expanded_solution = expanded_solution[v_policy_idx] # (N_STATES * N_GENS, SOLUTION_LEN) == [SOLUTION(S1), SOLUTION(S1), ..., SOLUTION(SN), SOLUTION(SN)]            
                neighbours_states = neighbours_states[v_policy_idx, :] # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]
                neighbors_policy_flatten = neighbors_policy_flatten[v_policy_idx] # (N_STATES * N_GEN) [POLICY_(A1(S1)), POLICY_(A2(S1)), ..., POLICY_(AN(SN))]
                expanded_parent_cumulative_policy = expanded_parent_cumulative_policy[v_policy_idx] # (N_GENS * N_STATES) [CUM(S1), CUM(S1), ..., CUM(SN), CUM(SN)]
                policy_scores = policy_scores[v_policy_idx] # (N_GENS * N_STATES) [CUM(S1) + LOG_POLICY_(A1(S1)), CUM(S1) + LOG_POLICY_(A2(S1)), ..., CUM(SN) + LOG_POLICY_(AN(SN))]
                # neighbours_hashes = neighbours_hashes[v_idx]
                v = v[v_policy_idx]
                p = p[v_policy_idx, :]
                value_policy_score = value_policy_score[v_policy_idx]

                if self.verbose:
                    print(f"{self.global_i}) P: {policy_scores[:3]}; P_MAX: {np.round(policy_score_max, 5)}")
                    print(f"{self.global_i}) V: {v[:3]}; V_MIN: {np.round(v_min, 5)}")
                    print(f"{self.global_i}) V+P: {value_policy_score[:3]}")
        
        self.value = v # (N_STATES) - one value for one state
        self.states = neighbours_states  # (N_STATES, STATE_SIZE)
        self.neighbors_policy = p # (N_STATES, N_GENS) - ~12 floats for one state
        
        self.solutions = torch.cat([expanded_solution, expanded_actions.unsqueeze(dim=1)], dim=1)
        self.parent_cumulative_policy = policy_scores

        if self.verbose:
            print(f"{self.global_i}) Processed count: {int_to_human(self.processed_count)}")
        if self.verbose:
            pass
            # print(f"{self.global_i}) v:", v[:3], "; policy_s:", policy_scores[:3] )        

        ######## ######## ######## ######## ########        

        if self.verbose:
            print(f"{self.global_i}) Iter time: {np.round(TimeContext.full_time,3)} sec")
        TimeContext.full_time = 0.0
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
        self.states = state.clone().to(self.device) # (N_STATES, STATE_SIZE)
        
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
            dtype=torch.int64,
            device=self.device
        ) # (N_STATES, N_LENGTH) - one path for each state

        # self.solution_lengths = torch.zeros((1)) # (N_STATES) - one float for one state
        self.parent_cumulative_value = torch.zeros((1), device=self.device) # (N_STATES) - one float for one state
        self.parent_cumulative_policy = torch.zeros((1), device=self.device) # (N_STATES) - one float for one state

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
                # if (len(search_result)) > 1:
                #     for i in range(len(search_result)):
                #         solution_index = search_result[i].item()
                #         solution = self.solutions[solution_index, :]
                #         print("SOL:", solution[1:].detach().tolist())

                solution_index = search_result[0].item()
                solution = self.solutions[solution_index, :]

                return solution[1:], self.processed_count
            
        return None, self.processed_count

def process_deepcube_dataset(
    report_path: str,
    model_path: str,
    search_mode: str, # value, policy, value_policy
    start_cube: int = 0,
    end_cubes: int = 100,
    verbose: bool = False,
    model_device = "cuda",
    is_state_dict_model: bool = True
):
    print(f"Search mode: {search_mode}")
    set_seed(0)
    deepcube_test = open_pickle("./assets/data/deepcubea/data_0.pkl")
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    generators = torch.tensor(game.actions, dtype=torch.int64)

    device = "cpu"
    if is_state_dict_model:
        model = Pilgrim(
            input_dim = 54, 
            hidden_dim1 = 5000, 
            hidden_dim2 = 1000, 
            num_residual_blocks = 4 
        ) # ~14M
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))    
        model = model.to(device)
    else:
        model = torch.load(model_path, map_location=model_device)
        model = model.to(model_device)
        # print(model)

    optimal_lens = []
    our_lens = []
    report = []
    for i in tqdm(range(start_cube, end_cubes)):
        state = torch.tensor(deepcube_test['states'][i], dtype=torch.int64).unsqueeze(0)
        opt_solution = deepcube_test['solutions'][i]

        optimal_lens.append(len(opt_solution))

        goal_state = torch.arange(0, 54, dtype=torch.int64)
        
        start = time.time()
        beam_search = BeamSearchMixExp(
            model=model,
            generators=torch.tensor(game.actions, dtype=torch.int64, device=device),
            num_steps=1000,
            policy_beam_width=-1,
            value_beam_width=-1,
            alpha=0.7,
            T=0.95,
            B=4,
            goal_state=goal_state,
            verbose=verbose,
            device=device,
            model_device=model_device
        )
        solution, processed_count = beam_search.search(state=state)

        end = time.time()

        duration = np.round(end - start, 3)

        if solution is None:
            solution = torch.tensor([], dtype=torch.int64, device=device)
            solution_len = -1
        else:
            solution_len = len(solution)

        record = {
            "i": i,
            "state": deepcube_test['states'][i],
            "duration_sec": duration,
            "solution_len": solution_len,            
            "solution": solution.detach().tolist(),
            "optimum_len": len(opt_solution),            
            "optimum": opt_solution,
            "processed_count": processed_count,
        }

        print(f"{i}] state:", state.shape)
        # print(f"{i}] optimum_len:", len(opt_solution))

        # print(f"{i}] solution_len:", len(solution), "path:", solution)
        print(f"{i}] solution_len:", solution_len)
        our_lens.append(solution_len)
        count_millions = np.round(processed_count / 10**6, 3)
        print(f"{i}] processed_count: {count_millions}M")
        print(f"{i}] duration: {duration} sec")
        print(f"{i}] our_lens:", our_lens)
        
        count_not_equal_minus_one = sum(1 for length in our_lens if length != -1)
        print(f"{i}] found soultions: {count_not_equal_minus_one}")
        
        # print(f"{i}] optimum mean:", np.round(np.mean(optimal_lens), 4))
        # print(f"{i}] our mean:", np.round(np.mean(our_lens), 4))

        report.append(record)

        if report_path is not None:
            save_pickle(report, report_path)
        print("\n")

if __name__ == "__main__":    
    # process_deepcube_dataset(
    #     report_path="./assets/reports/value_2B_800K_search_value.pkl",
    #     model_mode = "value",
    #     search_mode = "value",
    #     count_cubes = 100
    # )

    # process_deepcube_dataset(
    #     report_path="./assets/reports/policy_2B_800K_search_policy.pkl",
    #     model_mode = "policy",
    #     search_mode = "policy",
    #     count_cubes = 100
    # )

    # process_deepcube_dataset(
    #     report_path="./assets/reports/value_policy_2B_800K_search_value.pkl",
    #     model_mode = "value_policy",
    #     search_mode = "value",
    #     count_cubes = 100
    # )    

    # process_deepcube_dataset(
    #     report_path="./assets/reports/value_policy_2B_800K_search_policy.pkl",
    #     model_mode = "value_policy",
    #     search_mode = "policy",
    #     count_cubes = 100
    # )

    # process_deepcube_dataset(
    #     report_path="./assets/reports/Cube3ResnetModel_value_policy_3_8B_14M_search_value_full.pkl",
    #     model_path = "./assets/models/Cube3ResnetModel_value_policy_3_8B_14M.pt",
    #     search_mode = "value",
    #     start_cube = 0,
    #     end_cubes = 1000,
    #     verbose = False,
    #     model_device = "cuda"
    # )

    # process_deepcube_dataset(
    #     report_path=None,
    #     model_path = "./assets/models/Cube3ResnetModel_value_policy_3_8B_14M.pt",
    #     search_mode = "value",
    #     start_cube = 0,
    #     end_cubes = 1,
    #     verbose = True,
    #     model_device = "mps"
    # )    

    process_deepcube_dataset(
        report_path=None,
        model_path = "./assets/models/pruning_finetune_Cube3ResnetModel_value_policy_3_8B_14M.pt",
        search_mode = "policy",
        start_cube = 0,
        end_cubes = 100,
        verbose = False,
        model_device = "mps",
        is_state_dict_model=False
    )        