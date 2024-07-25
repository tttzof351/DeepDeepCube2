from accelerate import Accelerator
import torch
import numpy as np
import time 

from datasets import Cube3Dataset2
from datasets import Cube3Dataset3 
from datasets import scrambles_collate_fn
from cube3_game import Cube3Game
from models import Pilgrim
from models import count_parameters

from utils import set_seed
from utils import int_to_human
from hyperparams import hp


# train on 76K items
def train_nn():
    set_seed(hp["train_seed"])
    accelerator = Accelerator()
    device = accelerator.device    
    print("Accelerator device:", str(device))

    game = Cube3Game("./assets/envs/qtm_cube3.pickle")    

    # training_dataset = Cube3Dataset2(
    #     n = hp["cube3_god_number"],
    #     N = 10,
    #     size = 1_000_000,
    #     generators = torch.tensor(game.actions, dtype=torch.int64),
    # )
    # training_dataloader = torch.utils.data.DataLoader(
    #     training_dataset, 
    #     batch_size=4096,
    #     shuffle=True, 
    #     num_workers=4,
    #     collate_fn=scrambles_collate_fn
    # )

    training_dataset = Cube3Dataset3(
        n = hp["cube3_god_number"],
        N = 10,
        size = 1_000_000,
        generators = torch.tensor(game.actions, dtype=torch.int64, device="mps"),
        device="mps"
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, 
        batch_size=256,
        shuffle=True, 
        num_workers=4 if str(device) == "cpu" else 0,
        collate_fn=scrambles_collate_fn
    )

    model = Pilgrim(
        # hidden_dim1 = 500, 
        # hidden_dim2  = 300, 
        # num_residual_blocks = 3, 
    )
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
    val_count = 1000

    best_val_score = float("inf")
    start = time.time()
    trainset_count = 0
    with accelerator.autocast():
        while True:
            for data in training_dataloader:
                optimizer.zero_grad()
                model.train()

                states, actions, targets = data
                
                trainset_count += states.shape[0]
                v_out, policy_out = model(states)
                
                mse_loss = mse_loss_function(input=v_out, target=targets)
                cs_loss = cros_entroy_loss_function(input=policy_out, target=actions)

                # loss = mse_loss + cs_loss
                # loss = cs_loss
                loss = mse_loss

                accelerator.backward(loss)
                optimizer.step()

                rmse_accum_loss += np.sqrt(mse_loss.item())
                cs_accum_loss += cs_loss.item()
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
                    torch.save(model.state_dict(), "./assets/models/Cube3ResnetModel_value_.pt")
                    print(f"{global_i}) Saved model!")

                    # model.eval()
                    # val_acc_rmse = 0
                    # val_count_batch = 0               
                    # with torch.no_grad():
                    #     for val_data in val_dataloader:
                    #         val_count_batch += 1
                    #         val_states, val_targets = val_data
                    #         val_targets = val_targets.unsqueeze(dim=1)

                    #         # print("val_states:", val_states.shape)
                    #         # print("val_targets:", val_targets.shape)
                    #         val_outputs = model(val_states)
                    #         # print("val_outputs:", val_outputs.shape)
                    #         val_loss = mse_loss(val_outputs, val_targets)                        
                    #         val_acc_rmse += np.sqrt(val_loss.item())

                    # val_acc_rmse = np.round(val_acc_rmse / val_count_batch, 4)
                    # print("==========================")
                    # print(f"{global_i}): val_rmse={val_acc_rmse}")

                    # if val_acc_rmse < best_val_score:
                    #     torch.save(model.state_dict(), "./assets/models/Cube3ResnetModel.pt")
                    #     best_val_score = val_acc_rmse
                    #     print(f"Saved model!")
                    # else:
                    #     print(f"Old model is best! val_acc_rmse={val_acc_rmse} > best_val_score={best_val_score}")

            #     break
            # break

if __name__ == "__main__":
    train_nn()
