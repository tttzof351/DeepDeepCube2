from accelerate import Accelerator
import torch
import numpy as np
import time 

from datasets import Cube3Dataset
from cube3_game import Cube3Game
from models import Pilgrim

# train on 76K items
def train_nn():
    game = Cube3Game("./assets/envs/qtm_cube3.pickle")

    training_dataset = Cube3Dataset(
        length=22, 
        permutations=game.actions, 
        n=10,
        size=100
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset, 
        batch_size=32,
        shuffle=True, 
        num_workers=2
    )
    
    model = Pilgrim()#.to_script()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # accelerator = Accelerator(mixed_precision="fp16")
    accelerator = Accelerator()
    device = accelerator.device    
    model, optimizer, training_dataloader = accelerator.prepare(
        model, optimizer, training_dataloader
    )

    mse_loss_function = torch.nn.MSELoss()
    cros_entroy_loss_function = torch.nn.CrossEntropyLoss()

    print("Accelerator device:", device)

    global_i = 0
    rmse_accum_loss = 0
    print_count = 1
    val_count = 1000

    best_val_score = float("inf")
    start = time.time()
    with accelerator.autocast():
        while True:
            for data in training_dataloader:
                optimizer.zero_grad()
                model.train()

                states, actions, targets = data
                
                states = states.view(-1, 54)
                targets = targets.view(-1, 1)
                actions = actions.view(-1)

                # print("actions:", actions.shape, actions.dtype, actions.device)
                # print("states:", states.shape, states.dtype, states.device)
                # print("targets:", targets.shape, targets.dtype, states.device)
                v_out, a_out = model(states)

                # print("a_out:", a_out.shape, a_out.dtype, a_out.device)

                # print("outputs:", outputs.shape, outputs.dtype, states.device)
                
                mse_loss = mse_loss_function(input=v_out, target=targets)
                cs_loss = cros_entroy_loss_function(input=a_out, target=actions)

                loss = mse_loss + cs_loss
                # print("loss:", loss.item())

                accelerator.backward(loss)
                optimizer.step()

                rmse_accum_loss += np.sqrt(mse_loss.item())
                global_i += 1
                
                if (global_i % print_count == 0):
                    end = time.time()
                    duration = np.round(end - start, 3)
                    av_rmse_accum_loss = np.round(rmse_accum_loss / print_count, 3)

                    print(f"{global_i}): train_rmse={av_rmse_accum_loss}, duration={duration} sec")
                    rmse_accum_loss = 0.0
                    start = time.time()

                if (global_i % val_count == 0):
                    torch.save(model.state_dict(), "./assets/models/Cube3ResnetModel.pt")
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
