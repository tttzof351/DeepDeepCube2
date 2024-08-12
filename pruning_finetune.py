import torch_pruning as tp

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
from models import Pilgrim
from models import count_parameters

from utils import set_seed
from utils import int_to_human
from hyperparams import hp

def pruning(
    model: torch.nn.Module,
    device: torch.device,
    pruning_ratio: float = 0.5,
    n_gens: int = 12,
    space_size: int = 54
):
    example_inputs = torch.arange(0, space_size, dtype=torch.int32).unsqueeze(0).to(device)
    
    # 1. Importance criterion
    imp = tp.importance.GroupNormImportance(p=2) # or GroupTaylorImportance(), GroupHessianImportance(), etc.

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and (m.out_features == n_gens or m.out_features == 1):
            ignored_layers.append(m) # DO NOT prune the final classifier!

    pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
        ignored_layers=ignored_layers,
    )

    # base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

    mode_count_before = int_to_human(count_parameters(model))
    pruner.step()
    mode_count_after = int_to_human(count_parameters(model))
    print(f"Modle count: {mode_count_before} -> {mode_count_after}")

    # macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
    return model
    
def pruning_finetune(
    model_in_path: str,
    model_out_path: str,
    logs_out_path: str,
    trainset_limit: int,     
    device: torch.device   
):
    set_seed(hp["finetune_seed"])    
    model = Pilgrim(
        input_dim = 54, 
        hidden_dim1 = 5000, 
        hidden_dim2 = 1000, 
        num_residual_blocks = 4 
    ) # ~14M
    batch_size = 16
    model.load_state_dict(torch.load(model_in_path, map_location=device))
    model = model.to(device)
    
    model = pruning(model, device=device)    

    game = Cube3Game("./assets/envs/qtm_cube3.pickle")    

    training_dataset = Cube3Dataset3(
        n = hp["cube3_god_number"],
        N = 400,
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
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-3)

    mse_loss_function = torch.nn.MSELoss()
    cros_entroy_loss_function = torch.nn.CrossEntropyLoss()

    global_i = 0
    rmse_accum_loss = 0.0
    cs_accum_loss = 0.0
    print_count = 1
    val_count = 100

    best_val_score = float("inf")
    
    os.makedirs(logs_out_path, exist_ok=True)
    logger = SummaryWriter(logs_out_path)      
    
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
                cs_loss = cros_entroy_loss_function(input=policy_out, target=actions.long())

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
                
                torch.save(model_out_path)
                print(f"{global_i}) Saved model!")

                if trainset_count > trainset_limit:
                    stop_train = True
                    break

        if stop_train:
            break        


if __name__ == "__main__":
    pruning_finetune(
        model_in_path="./assets/models/Cube3ResnetModel_value_policy_3_8B_14M.pt",
        model_out_path="./assets/models/pruning_finetune_Cube3ResnetModel_value_policy_3_8B_14M.pt",
        logs_out_path="./assets/logs/pruning_finetune_Cube3ResnetModel_value_policy_3_8B_14M",
        device="cpu",
        trainset_limit=2_000_000_000
    )
        