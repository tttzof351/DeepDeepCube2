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
from models import Pilgrim, PilgrimTransformer, PilgrimSimple, PilgrimCNN, PilgrimMLP2
from models import count_parameters

from utils import set_seed
from utils import int_to_human
from hyperparams import hp


# train on 76K items
def train_nn(
    model: torch.nn.Module,
    model_path = None,
    log_path = None,    
    N = 400,
    trainset_limit = 2_000_000_000,
    device = "cpu"
):
    set_seed(hp["train_seed"])
    print("Device:", str(device))

    # model = Pilgrim(
    #     input_dim = 54, 
    #     hidden_dim1 = 5000, 
    #     hidden_dim2 = 1000, 
    #     num_residual_blocks = 4 
    # ) # ~14M
    batch_size = 16

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
    val_count = 1000

    best_val_score = float("inf")

    if log_path is not None:        
        os.makedirs(log_path, exist_ok=True)
        logger = SummaryWriter(log_path)  
    else:
        logger = SummaryWriter(f"/tmp")  
    
    start = time.time()
    trainset_count = 0
    stop_train = False
    
    use_amp = str(device) == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    while True:
        for data in training_dataloader:            
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                model.train()
                optimizer.train()                

                states, actions, targets = data
                
                states = states.to(device)
                actions = actions.to(device)
                targets = targets.to(device)
                
                trainset_count += states.shape[0]
                v_out, policy_out = model(states)
                
                mse_loss = mse_loss_function(input=v_out, target=targets)
                cs_loss = torch.tensor(-1.0)#cros_entroy_loss_function(input=policy_out, target=actions.long())

                loss = mse_loss #+ cs_loss

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

            scaler.step(optimizer)
            scaler.update()
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
                if model_path is not None:
                    torch.save(model.state_dict(), model_path)                
                    print(f"{global_i}) Saved model!")

                if trainset_count > trainset_limit:
                    stop_train = True
                    break

        if stop_train:
            break        

if __name__ == "__main__":
    # model = PilgrimTransformer(
    #     space_size = 54,
    #     n_gens = 12,
    #     d_model = 256,
    #     nhead = 4,
    #     num_layers = 4
    # )
    # model = torch.compile(model)
    # N = 1

    # model = Pilgrim(
    #     input_dim = 54, 
    #     hidden_dim1 = 5000, 
    #     hidden_dim2 = 1000, 
    #     num_residual_blocks = 4 
    # ) # ~14M    
    # N = 400

    # model = PilgrimCNN()
    # N = 1

    model = PilgrimMLP2()
    N = 10

    train_nn(
        model = model,
        model_path = "./assets/models/mlp2_value.pt",
        log_path = "./assets/logs/mlp2_value",
        N = N,
        trainset_limit = 4_000_000_000,
        device = "cpu"
    )
