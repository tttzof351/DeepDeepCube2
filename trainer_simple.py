import os 
import sys

from contextlib import nullcontext
import schedulefree
from schedulefree import ScheduleFreeWrapper

import torch
from torch.utils.tensorboard import SummaryWriter


import numpy as np
import time 

from g_datasets import Cube3Dataset3 
from g_datasets import scrambles_collate_fn
from cube3_game import Cube3Game
from models import PilgrimSimple
from models import count_parameters

from utils import set_seed
from utils import int_to_human
from hyperparams import hp


# train on 76K items
def train_nn(
    mode: str, # value, policy, value_policy,
    model_name: str = None,
    N = 100,
    trainset_limit = 2_000_000_000,
    device = "cpu"
):
    set_seed(hp["train_seed"])
    # accelerator = Accelerator(
    #     mixed_precision  = None #"fp16" if torch.cuda.is_available() else None
    # )
    # device = "cpu" #accelerator.device    
    
    print("Device:", str(device))

    # model = Pilgrim(
    #     hidden_dim1 = 500, 
    #     hidden_dim2  = 300, 
    #     num_residual_blocks = 3,    
    # ) # 800K
    # batch_size = 32

    model = PilgrimSimple()
    batch_size = 4

    model.to(device)


    game = Cube3Game("./assets/envs/qtm_cube3.pickle")    

    training_dataset = Cube3Dataset3(
        n = hp["cube3_god_number"],
        N = N,
        size = 1_000_000,
        generators = torch.tensor(game.actions, dtype=torch.int64, device=device),
        device=device
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=4 if str(device) == "cpu" else 0,
        collate_fn=scrambles_collate_fn
    )
               
    print("Count parameters:", int_to_human(count_parameters(model)))
    # base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # optimizer = ScheduleFreeWrapper(
    #     base_optimizer, momentum=0.9, weight_decay_at_y=0.1
    # )
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-3)

    mse_loss_function = torch.nn.MSELoss()
    cros_entroy_loss_function = torch.nn.CrossEntropyLoss()

    global_i = 0
    rmse_accum_loss = 0.0
    cs_accum_loss = 0.0
    print_count = 10
    val_count = 100

    best_val_score = float("inf")

    os.makedirs(f"./assets/logs/{mode}", exist_ok=True)
    if model_name is not None:
        logger = SummaryWriter(f"./assets/logs/{model_name}")  
    else:
        logger = SummaryWriter(f"./assets/logs/{mode}")  
    
    start = time.time()
    trainset_count = 0
    stop_train = False
    
    use_amp = str(device) == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if use_amp:
        ctx = torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp)
    else:
        ctx = nullcontext()

    while True:
        for data in training_dataloader:            
            with ctx:
                model.train()
                optimizer.train()                

                states, actions, targets = data
                
                states = states.to(device)
                actions = actions.to(device)
                targets = targets.to(device)
                
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

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()

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
                if model_name is not None:
                    model_path = f"./assets/models/{model_name}.pt"
                else:
                    model_path = f"./assets/models/Cube3ResnetModel_{mode}.pt"
                
                torch.save(model.state_dict(), model_path)
                print(f"{global_i}) Saved model!")

                if trainset_count > trainset_limit:
                    stop_train = True
                    break

        if stop_train:
            break        

if __name__ == "__main__":
    train_nn(
        mode = "value_policy",
        model_name = "Simple_value_policy_8B_14M",
        N = 10,
        trainset_limit = 1_000_000_000,
        device = "cpu"
    )
