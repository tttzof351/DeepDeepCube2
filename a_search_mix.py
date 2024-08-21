import numpy as np
import torch

from contextlib import nullcontext

from utils import open_pickle

from models import Pilgrim, PilgrimTransformer, PilgrimSimple, PilgrimCNN, PilgrimMLP2

from tqdm import tqdm
from cube3_game import Cube3Game
from models import Pilgrim
from g_datasets import get_torch_scrambles_3
from utils import set_seed
from utils import save_pickle
from utils import TimeContext
from utils import int_to_human

class AStarVector:
    def __init__(
        self,
        model: torch.nn.Module,
        num_steps: int, 
        b_exp: int,
        b_keep: int,
        temperature: float,
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
        
        self.num_steps = num_steps
        self.b_exp = b_exp
        self.b_keep = b_keep
        self.temperature = temperature
        self.generators = generators.to(self.device)
        self.n_gens = generators.size(0)
        self.state_size = generators.size(1)
        self.hash_vec = torch.randint(0, 1_000_000_000_000, (self.state_size,), device=self.device)
        self.goal_state = goal_state.to(self.device)
        self.verbose = verbose
        self.processed_count = 0

        self.model.to(self.model_device)
        self.model.eval()

        if self.use_amp:
            self.ctx = torch.autocast(device_type=self.model_device, dtype=torch.float16, enabled=self.use_amp)
        else:
            self.ctx = nullcontext()        

    def get_hashes(self, states: torch.Tensor):
        return torch.sum(self.hash_vec * states.int(), dim=1)

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

        values = values.to(self.device)#.cpu()
        policy = torch.softmax(policy.float(), dim=1).to(self.device)#.cpu()

        return values, policy
    
    def update_greedy_step(self):
        # print("h:", self.h)
        # print("g:", self.g)

        f = self.h + self.g * 1.0
        
        if self.temperature > 0.0:
            # print("SOFTMAX!")
            f_prob = torch.softmax(-f.double() / self.temperature, dim=0)

            B = self.b_exp + self.b_keep
            if f_prob.shape[0] < B:
                B = f_prob.shape[0]
            sorted_f_ids = f_prob.multinomial(num_samples=B, replacement=False)

        else:
            sorted_f_ids = torch.argsort(f, descending=False)
        
        # print(f"{self.global_i}) sorted_f_ids_canonical: {sorted_f_ids_canonical[:3]}; prob: {f_prob[sorted_f_ids_canonical[:3]]}")
        # print(f"{self.global_i}) sorted_f_ids: {sorted_f_ids[:3]}; prob: {f_prob[sorted_f_ids[:3]]}" )

        expend_ids = sorted_f_ids[0:self.b_exp]
        keep_ids = sorted_f_ids[self.b_exp:(self.b_exp + self.b_keep)]

        s_expended = self.states[expend_ids]
        s_keep = self.states[keep_ids]

        g_expend = self.g[expend_ids]
        g_keep = self.g[keep_ids]

        h_expended = self.h[expend_ids] # from neural network, see below
        h_keep = self.h[keep_ids]

        solutions_expend = self.solutions[expend_ids, :]
        solutions_keep = self.solutions[keep_ids, :]

        ############

        n_states = s_expended.shape[0]
        state_size = s_expended.shape[1]

        s_expended = s_expended.unsqueeze(dim=1).expand(
            n_states,
            self.n_gens,
            state_size,
        ).reshape(
            self.n_gens * n_states,
            state_size
        ).to(self.device) # (N_STATES * N_GENS, STATE_SIZE) == [S1, S1, ..., SN, SN]

        expanded_actions = torch.arange(0, self.n_gens).unsqueeze(dim=0).expand(
            n_states,
            self.n_gens
        ).reshape(
            n_states * self.n_gens
        ).to(self.device) # (N_GENS * STATE_SIZE) == [A1, A2, ..., A1, A2]

        s_expended = torch.gather(
            input=s_expended,
            dim=1,
            index=self.generators[expanded_actions, :]
        ).to(self.device) # (N_STATES * N_GENS, STATE_SIZE) [A1(S1), A2(S1), ..., AN(SN)]

        g_expend = 1 + g_expend.unsqueeze(dim=1).expand(
            n_states,
            self.n_gens
        ).reshape(
            self.n_gens * n_states
        ) # (N_STATES * N_GENS) == [G1+1, G1+1, G1+1, ..., GN+1, GN+1, GN+1]

        ###########
        
        s_hashes = self.get_hashes(s_expended)
        hashed_sorted, hashed_idx = torch.sort(s_hashes)
        unique_hahes_mask_2 = torch.cat((torch.tensor([True], device=self.device), hashed_sorted[1:] - hashed_sorted[:-1] > 0))
        hashed_idx = hashed_idx[unique_hahes_mask_2]
        
        s_expended = s_expended[hashed_idx]
        g_expend = g_expend[hashed_idx]
        expanded_actions = expanded_actions[hashed_idx]

        ############

        solution_depth = solutions_expend.shape[1]
        solutions_expend = solutions_expend.unsqueeze(dim=1).expand(
            n_states,
            self.n_gens,
            solution_depth
        ).reshape(
            self.n_gens * n_states,
            solution_depth
        ).to(self.device) # (N_STATES * N_GENS, SOLUTION_LEN) == [SOLUTION(S1), SOLUTION(S1), ..., SOLUTION(SN), SOLUTION(SN)]
        
        solutions_expend = solutions_expend[hashed_idx, :] # ????

        solutions_expend = torch.cat([solutions_expend, expanded_actions.unsqueeze(dim=1)], dim=1)        

        void_step = torch.full(
            size=[solutions_keep.shape[0], 1], # (different solutions, path of action)
            fill_value=-1,
            dtype=torch.int64,
            device=self.device
        ) # (N_STATES, N_LENGTH) - one path for each state
        
        solutions_keep = torch.cat([
            solutions_keep, 
            void_step
        ], dim=1)

        ############

        if self.verbose :
            print(f"{self.global_i}) predict for: ", s_expended.shape[0])
        h_expended, _ = self.predict(s_expended)

        ############

        self.h = torch.cat([h_expended, h_keep], dim=0)
        self.g = torch.cat([g_expend, g_keep], dim=0)

        if solutions_keep.shape[0] > 1:
            self.solutions = torch.cat([solutions_expend, solutions_keep], dim=0)
        else:
            self.solutions = solutions_expend

        self.states = torch.cat([s_expended, s_keep], dim=0)

        # if self.verbose:
            # print(f"{self.global_i}) g_mean: {np.round( torch.mean(self.g).item(),3)}; h_mean: {np.round( torch.mean(self.h).item(),3)}")
            # print(f"{self.global_i}) g_min: {torch.min(self.g)}; g_max: {torch.max(self.g)}; g_mean: {np.round(torch.mean(self.g).item(), 3)};, g_size: {self.g.shape[0]}")
            # print(f"{self.global_i}) h_min: {np.round(torch.min(self.h).item(), 3)}; h_max: {np.round(torch.max(self.h).item(), 3)}; h_mean: {np.round(torch.mean(self.h).item(), 3)}; h_size: {np.round(self.h.shape[0], 3)}")

        pass

    def search(
        self,
        state: torch.Tensor
    ):
        if len(state.shape) < 2:
            state = state.unsqueeze(0)

        self.model.eval()
        self.states = state.clone().to(self.device) # (N_STATES, STATE_SIZE)
        self.g = torch.zeros((1), device=self.device) # (N_STATES) - one float for one state
        self.h, _ = self.predict(self.states) # (N_STATES)

        self.solutions = torch.full(
            size=[1, 1], # (different solutions, path of action)
            fill_value=-1,
            dtype=torch.int64,
            device=self.device
        ) # (N_STATES, N_LENGTH) - one path for each state

        self.global_i = 0

        for j in range(self.num_steps):
            self.update_greedy_step()
            search_result = (self.states == self.goal_state).all(dim=1).nonzero(as_tuple=True)[0]
            
            if (len(search_result) > 0):
                # print("Found!", search_result)
                # for i in range(len(search_result)):
                #     solution_index = search_result[i].item()
                #     solution = self.solutions[solution_index, :]
                    # print("solution:", solution)

                solution_index = search_result[0].item()
                solution = self.solutions[solution_index, :]

                return solution[solution != -1], self.processed_count

            self.global_i += 1            

        return [], self.processed_count
    
def test_a_star():
    set_seed(0)
    deepcube_test = open_pickle("./assets/data/deepcubea/data_0.pkl")
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    generators = torch.tensor(game.actions, dtype=torch.int64)

    device = "cpu"
    model_device = "mps"

    model = Pilgrim(
        input_dim = 54, 
        hidden_dim1 = 5000, 
        hidden_dim2 = 1000, 
        num_residual_blocks = 4 
    ) # ~14M

    model.load_state_dict(
        torch.load(
            "./assets/models/Cube3ResnetModel_value_policy_3_8B_14M.pt",
            map_location=model_device)
    )
    model = model.to(model_device)
    goal_state = torch.arange(0, 54, dtype=torch.int64)

    our_lens = []
    for i in tqdm(range(0, 1)):
        with TimeContext(f"{i}] Execution time:", True):            
            state = torch.tensor(deepcube_test['states'][i], dtype=torch.int64)#.unsqueeze(0)

            state = state.unsqueeze(0)

            path_finder = AStarVector(
                model=model,
                generators=generators,
                num_steps=10_000,
                b_exp=100_000,
                b_keep=100_000,
                temperature=0.0,
                goal_state=goal_state,
                verbose=True,
                device=device,
                model_device=model_device
            )
            
            opt_solution = deepcube_test['solutions'][i]

            solution, processed_count = path_finder.search(state)
            our_lens.append(len(solution))

            print(f"{i}] optimum_len:", len(opt_solution))
            print(f"{i}] solution len:", len(solution))
            print(f"{i}] mean len:", np.round(np.mean(our_lens), 3))
            print(f"{i}] solution:", solution)
            print(f"{i}] processed_count:", int_to_human(processed_count))

def sample_solutions():
    deepcube_test = open_pickle("./assets/data/deepcubea/data_0.pkl")
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    generators = torch.tensor(game.actions, dtype=torch.int64)

    device = "cpu"
    model_device = "cpu"

    model = Pilgrim(
        input_dim = 54, 
        hidden_dim1 = 5000, 
        hidden_dim2 = 1000, 
        num_residual_blocks = 4 
    ) # ~14M

    model.load_state_dict(
        torch.load(
            "./assets/models/Cube3ResnetModel_value_policy_3_8B_14M.pt",
            map_location=model_device)
    )
    model = model.to(model_device)
    goal_state = torch.arange(0, 54, dtype=torch.int64)
    
    path = []
    while True:
        n_stats = []
        for n in range(len(generators)):
            lens = []
            for seed in range(10):
                set_seed(seed + 10 * n)
                state = torch.tensor(deepcube_test['states'][1], dtype=torch.int64)#.unsqueeze(0)            
                for a in path:
                    state = state[generators[a]]
                
                is_goal = (state == goal_state).all().item()
                if is_goal:
                    print(f"Len:", len(path))
                    print(f"Path: {path}")
                    return
                
                state = state[generators[n]]
                state = state.unsqueeze(0)

                path_finder = AStarVector(
                    model=model,
                    generators=generators,
                    num_steps=10_000,
                    b_exp=100,
                    b_keep=100,
                    temperature=1e-1,
                    goal_state=goal_state,
                    verbose=False,
                    device=device,
                    model_device=model_device
                )
                
                solution, _ = path_finder.search(state)
                if len(solution) == 1:
                    path.append(n)
                    print("Solution:", solution)
                    print(f"Len:", len(path))
                    print(f"Path: {path}")
                    
                    return
                
                lens.append(len(solution))        
            # print(f"n: {n}; mean: {np.mean(lens)}")
            n_stats.append(np.mean(lens))
        
        action = np.argsort(n_stats)[0]
        print("Element of path!: ", action)
        path.append(action)

def find_best_temperature():
    set_seed(0)
    deepcube_test = open_pickle("./assets/data/deepcubea/data_0.pkl")
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    generators = torch.tensor(game.actions, dtype=torch.int64)

    device = "cpu"
    model_device = "cpu"

    model = Pilgrim(
        input_dim = 54, 
        hidden_dim1 = 5000, 
        hidden_dim2 = 1000, 
        num_residual_blocks = 4 
    ) # ~14M

    model.load_state_dict(
        torch.load(
            "./assets/models/Cube3ResnetModel_value_policy_3_8B_14M.pt",
            map_location=model_device)
    )
    model = model.to(model_device)
    goal_state = torch.arange(0, 54, dtype=torch.int64)

    TS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    BS = [128, 256, 512, 1024, 2048, 4096, 8192]

    for b in BS:
        mean_lens = []
        for t in TS:
            our_lens = []
            with TimeContext(f"b={b}; t={t}] Execution time:", True):
                for i in tqdm(range(0, 10)):
                    state = torch.tensor(deepcube_test['states'][i], dtype=torch.int64)#.unsqueeze(0)

                    state = state.unsqueeze(0)

                    path_finder = AStarVector(
                        model=model,
                        generators=generators,
                        num_steps=10_000,
                        b_exp=b,
                        b_keep=0,
                        temperature=t,
                        goal_state=goal_state,
                        verbose=False,
                        device=device,
                        model_device=model_device
                    )
                    
                    # opt_solution = deepcube_test['solutions'][i]

                    solution, processed_count = path_finder.search(state)
                    our_lens.append(len(solution))        
            mean_lens.append(np.round(np.mean(our_lens), 4))
        best_id = np.argsort(mean_lens)[0]
        best_t = TS[best_id]
        best_mean = mean_lens[best_id]
        print(f"b={b}; means={mean_lens}; best_mean={best_mean}, best_t={best_t}")

def find_best_keep():
    set_seed(0)
    deepcube_test = open_pickle("./assets/data/deepcubea/data_0.pkl")
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    generators = torch.tensor(game.actions, dtype=torch.int64)

    device = "cpu"
    model_device = "mps"

    model = Pilgrim(
        input_dim = 54, 
        hidden_dim1 = 5000, 
        hidden_dim2 = 1000, 
        num_residual_blocks = 4 
    ) # ~14M

    model.load_state_dict(
        torch.load(
            "./assets/models/Cube3ResnetModel_value_policy_3_8B_14M.pt",
            map_location=model_device)
    )
    model = model.to(model_device)
    goal_state = torch.arange(0, 54, dtype=torch.int64)

    # B_KEEP = [0, 12, 120, 1200, 12000]
    # B_EXP = [4096]

    # B_KEEP = [0, 12, 120, 1200, 12000]
    # B_EXP = [2048, 4096, 8192]

    # B_KEEP = [0]
    # B_EXP = [4096]

    B_KEEP = [4096]
    B_EXP = [20]

    for b in B_EXP:
        mean_lens = []
        for b_keep in B_KEEP:
            our_lens = []
            b_exp = b -int(b_keep / 12)
            with TimeContext(f"b_exp={b_exp}; b_keep={b_keep}] Execution time:", True):
                for i in tqdm(range(0, 100)):
                    state = torch.tensor(deepcube_test['states'][i], dtype=torch.int64)#.unsqueeze(0)

                    state = state.unsqueeze(0)

                    path_finder = AStarVector(
                        model=model,
                        generators=generators,
                        num_steps=10_000,
                        b_exp=b_exp,
                        b_keep=b_keep,
                        temperature=0.0,
                        goal_state=goal_state,
                        verbose=False,
                        device=device,
                        model_device=model_device
                    )
                    
                    # opt_solution = deepcube_test['solutions'][i]

                    solution, processed_count = path_finder.search(state)
                    our_lens.append(len(solution))        
            mean_lens.append(np.round(np.mean(our_lens), 4))
        best_id = np.argsort(mean_lens)[0]
        best_b_keep = B_KEEP[best_id]
        best_b_exp = b - int(best_b_keep / 12)
        best_mean = mean_lens[best_id]
        print(f"b={b}; means={mean_lens}; best_mean={best_mean}, best_b_exp={best_b_exp}; best_b_keep={best_b_keep}")


if __name__ == "__main__": 
    test_a_star()
    # sample_solutions()
    # find_best_temperature()
    # find_best_keep()