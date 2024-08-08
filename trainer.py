import os 
import sys

from accelerate import Accelerator

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time 

from g_datasets import Cube3Dataset3 
from g_datasets import scrambles_collate_fn
from cube3_game import Cube3Game
from models import Pilgrim
from models import count_parameters

from utils import set_seed
from utils import int_to_human
from hyperparams import hp


# train on 76K items
def train_nn(
    mode: str, # value, policy, value_policy
    trainset_limit = 2_000_000_000
):
    set_seed(hp["train_seed"])
    accelerator = Accelerator(
        mixed_precision  = None #"fp16" if torch.cuda.is_available() else None
    )
    device = accelerator.device    
    
    print("Accelerator device:", str(device))

    game = Cube3Game("./assets/envs/qtm_cube3.pickle")    

    training_dataset = Cube3Dataset3(
        n = hp["cube3_god_number"],
        N = 400,
        size = 1_000_000,
        generators = torch.tensor(game.actions, dtype=torch.int64, device=accelerator.device),
        device=accelerator.device
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, 
        batch_size=32,
        shuffle=True, 
        num_workers=4 if str(device) == "cpu" else 0,
        collate_fn=scrambles_collate_fn
    )

    model = Pilgrim(
        hidden_dim1 = 500, 
        hidden_dim2  = 300, 
        num_residual_blocks = 3,    
    ) # 800K

    # model = Pilgrim(
    #     input_dim = 54, 
    #     hidden_dim1 = 5000, 
    #     hidden_dim2 = 1000, 
    #     num_residual_blocks = 4 
    # ) # ~14M
               
    print("Count parameters:", count_parameters(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # accelerator = Accelerator(mixed_precision="fp16")
    model, optimizer, training_dataloader = accelerator.prepare(
        model, optimizer, training_dataloader
    )

    mse_loss_function = torch.nn.MSELoss()
    cros_entroy_loss_function = torch.nn.CrossEntropyLoss()

    global_i = 0
    rmse_accum_loss = 0.0
    cs_accum_loss = 0.0
    print_count = 1
    val_count = 100

    best_val_score = float("inf")

    os.makedirs(f"./assets/logs/{mode}", exist_ok=True)
    logger = SummaryWriter(f"./assets/logs/{mode}")  
    start = time.time()
    trainset_count = 0
    stop_train = False

    with accelerator.autocast():
        while True:
            for data in training_dataloader:
                optimizer.zero_grad()
                model.train()

                states, actions, targets = data
                
                trainset_count += states.shape[0]
                v_out, policy_out = model(states)
                
                mse_loss = mse_loss_function(input=v_out, target=targets)
                cs_loss = cros_entroy_loss_function(input=policy_out, target=actions.long())

                if mode == "value":
                    loss = mse_loss
                elif mode == "policy":
                    loss = cs_loss
                elif mode == "value_policy":
                    loss = mse_loss + cs_loss
                else:
                    raise f"Incorreect mode: {mode}"                

                accelerator.backward(loss)
                optimizer.step()

                rmse_accum_loss += np.sqrt(mse_loss.item())
                cs_accum_loss += cs_loss.item()

                logger.add_scalar("Loss/rmse", np.sqrt(mse_loss.item()), global_step=global_i)
                logger.add_scalar("Loss/cross_entropy", cs_loss.item(), global_step=global_i)

                global_i += 1
                
                if (global_i % print_count == 0):
                    end = time.time()
                    duration = np.round(end - start, 3)
                    av_rmse_accum_loss = np.round(rmse_accum_loss / print_count, 3)
                    av_cs_accum_loss = np.round(cs_accum_loss / print_count, 3)

                    print(f"{global_i}): rmse={av_rmse_accum_loss}; cross_e={av_cs_accum_loss}, duration={duration} sec; trainset={int_to_human(trainset_count)}")
                    rmse_accum_loss = 0.0
                    cs_accum_loss = 0.0
                    start = time.time()

                if (global_i % val_count == 0):
                    os.makedirs("./assets/models/", exist_ok=True)
                    torch.save(model.state_dict(), f"./assets/models/Cube3ResnetModel_{mode}.pt")
                    print(f"{global_i}) Saved model!")

                    if trainset_count > trainset_limit:
                        stop_train = True
                        break

            if stop_train:
                break        

if __name__ == "__main__":
    train_nn(mode = "value")
