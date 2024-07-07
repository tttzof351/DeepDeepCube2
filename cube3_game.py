import numpy as np
import pickle as pkl
from tqdm import tqdm

from numba.experimental import jitclass
import numba

spec = [
    ('qtm_path', numba.types.unicode_type),          
    ('action_id', numba.int32), 
    ('state', numba.int32[:])
]

# @jitclass(spec)
class Cube3Game:
    def __init__(self, qtm_path: str):        
        with open(qtm_path, "rb") as f:
            env_dict = pkl.load(f)
            self.actions: np.array = np.array(env_dict['actions']).astype(np.int32)
            self.names: list[str] = env_dict["names"]
            
            self.id_to_name = { i:n for i, n in enumerate(self.names) }
            self.name_to_id = { n:i for i, n in enumerate(self.names) }

        self.action_size = self.actions.shape[0]
        self.direct_action_size = int(self.action_size / 2)

        self.space_size = self.actions.shape[1]

        self.initial_state = np.arange(0, self.space_size).astype(np.int32)

    def apply_action(self, state, action):        
        return state[self.actions[action]].astype(np.int32)
    
    def reverse_action(self, action):
        if action >= self.direct_action_size:
            return action - self.direct_action_size
        else:
            return self.direct_action_size + action

    def is_goal(self, state):
        return np.array_equal(state, self.initial_state)
    
    def to_nn(self, state):
        return (state / 9.0).astype(np.int32)