import numpy as np
from utils import array_wyhash
from cube3_game import Cube3Game

if __name__ == "__main__":
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")
    print(game.names)