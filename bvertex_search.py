import time

import torch
import torch.nn.functional as F
import numpy as np

from utils import open_pickle

from cube3_game import Cube3Game
from models import Pilgrim

class Node:
    def __init__(self, state: torch.Tensor):
        self.state = state

        self.g = 0.0
        self.log_policy = 0.0
        self.log_value = 0.0

class VertexSearch:
    def __init__(
        self,
        model: torch.nn.Module,
        generators: torch.Tensor,
        goal_state: torch.Tensor,
        device: torch.device
    ):  
        self.model = model
        self.generators = generators
        self.n_gens = generators.size(0)
        self.state_size = generators.size(1)
        self.device = device
        self.hash_vec = torch.randint(0, 1_000_000_000_000, (self.state_size,))
        self.goal_state = goal_state

        self.model.eval()
        self.model.to(device)

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
            print(f"Processed: {count_millions}M")
            self.printed_count = self.processed_states_count

        values = values.cpu()
        policy = torch.softmax(policy, dim=1).cpu()

        return values, policy
    
    # def get_neighbors(
    #     self, 
    #     states: torch.Tensor
    # ) -> torch.Tensor:
    #     expanded_states = states.unsqueeze(1).expand(states.size(0), self.n_gens, self.state_size)        
    #     indexes = self.generators.unsqueeze(0).expand(states.size(0), self.n_gens, self.state_size)        
        
    #     new_states = torch.gather(input=expanded_states, dim=2, index=indexes)

    #     return new_states

    def do_search_step(self):
        # states = torch.cat([n.state for n in self.open])
        neibhbors = []
        [n.state[:, g] for g in self.generators] for n in self.open]
        print("neibhbors:", neibhbors)

        # values, policy = self.predict_values(neibhbors)

        # print("states:", states.shape)        

        self.global_i += 1

        if self.global_i > 2:
            exit()

    def search(
        self,
        state: torch.Tensor
    ):  
        print("state:", state.shape)

        self.global_i = 0
        self.processed_states_count = 0   
        self.printed_count = 0

        self.open = [ Node(state) ]

        while len(self.open) > 0:
            self.do_search_step()


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

    vertex_search = VertexSearch(
        model = model,
        generators = generators,
        goal_state = goal_state,
        device = "mps"
    )    

    vertex_search.search(state)