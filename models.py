import numpy as np
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import int_to_human
from typing import Dict,Tuple,Optional,List


class ResidualBlock(nn.Module):
    def __init__(
            self, 
            hidden_dim: int,
            dropout_rate: float = 0.1
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
    

class FFBlock(nn.Module):
    def __init__(
        self, 
        hidden_in_dim: int,
        hidden_out_dim: int
    ) -> None:
        super(FFBlock, self).__init__()
        self.hidden_in_dim = hidden_in_dim
        self.hidden_out_dim = hidden_out_dim

        self.ln1 = nn.LayerNorm(hidden_in_dim)
        self.fc1 = self.xavier(nn.Linear(hidden_in_dim, hidden_out_dim))
        self.relu = nn.ReLU()
        self.fc2 = self.xavier(nn.Linear(hidden_out_dim, hidden_out_dim))
        self.ln2 = nn.LayerNorm(hidden_out_dim)

    def xavier(self, layer):
        torch.nn.init.xavier_normal_(layer.weight)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.ln1(x)        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.ln2(out)        
        out = self.fc2(out)  
        if self.hidden_in_dim == self.hidden_out_dim:
            out += residual
        out = self.relu(out)
        return out
    

class CNNResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding='same', 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding='same', 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=(3, 3), 
                    padding='same', 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample else x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)

        return x

class Pilgrim(nn.Module):
    def __init__(
        self, 
        input_dim: int = 54, 
        hidden_dim1: int = 400, 
        hidden_dim2: int = 200, 
        num_residual_blocks: int = 2, 
        output_dim: int = 1, 
        dropout_rate: float = 0.1,
        n_gens: int = 12
    ) -> None:
        super(Pilgrim, self).__init__()
        self.hd1 = hidden_dim1
        self.hd2 = hidden_dim2
        self.nrd = num_residual_blocks
        
        self.input_layer = nn.Linear(input_dim * 6, hidden_dim1)
        self.hidden_layer = nn.Linear(hidden_dim1, hidden_dim2)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim2, dropout_rate) for _ in range(num_residual_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim2, output_dim)
        self.output_probs_layer = nn.Linear(hidden_dim2, n_gens)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (x / 9).long()

        x = F.one_hot(x.long(), 6)
        x = x.float()
        x = x.view(-1, 54 * 6)

        x = self.relu(self.input_layer(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(self.hidden_layer(x))
        x = self.bn2(x)
        x = self.dropout(x)
        for layer in self.residual_blocks:
            x = layer(x)
        
        out = self.output_layer(x)

        probs_out = self.output_probs_layer(x)

        return out, probs_out

    def to_script(self):
        model = torch.jit.script(self)
        return model
    
    def traced_model(self):
        model = torch.jit.trace(self, torch.randint(low=0, high=54, size=(2, 54)))
        return model

class PilgrimTransformer(nn.Module):
    def __init__(
            self,
            space_size = 54,
            n_gens = 12,
            d_model = 64,
            nhead = 4,
            num_layers = 4
        ):
        super(PilgrimTransformer, self).__init__()

        self.space_size = space_size
        self.n_gens = n_gens
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.color_devider = int(space_size/6) # 9

        self.encoder_transformer = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model, 
                nhead=self.nhead,
                dropout=0.0,
                batch_first=True, 
                norm_first=False,
            ),
            enable_nested_tensor=True,
            norm=nn.LayerNorm(self.d_model),
            num_layers=self.num_layers
        )

        self.input_embedding = nn.Embedding(
            num_embeddings=6, 
            embedding_dim=self.d_model
        )

        self.pos_encoding = nn.Embedding(
            num_embeddings=self.space_size, 
            embedding_dim=self.d_model
        )

        self.output_value_layer = nn.Linear(self.d_model * self.space_size, 1)
        self.output_probs_layer = nn.Linear(self.d_model * self.space_size, self.n_gens)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (x / self.color_devider).long()

        x = self.input_embedding(x)

        pos_codes = self.pos_encoding(
            torch.arange(54, device=x.device),            
        )

        x += pos_codes
        x = self.encoder_transformer(x)
        
        x = x.view(-1, self.d_model * self.space_size)
        value = self.output_value_layer(x)
        policy = self.output_probs_layer(x)

        return value, policy
    
class PilgrimSimple(nn.Module):
    def __init__(
        self,
        space_size = 54,
        n_gens = 12,
        d_model = 256,
        num_layers = 4
    ):
        super(PilgrimSimple, self).__init__()
        self.space_size = space_size
        self.n_gens = n_gens
        self.d_model = d_model
        self.num_layers = num_layers

        self.color_devider = int(space_size/6) # 9

        self.input_linear = nn.Linear(
            self.space_size,
            self.d_model
        )

        self.liear_blocks = nn.ModuleList([
            # nn.Linear(self.d_model, self.d_model) for _ in range(num_layers)
            ResidualBlock(self.d_model, dropout_rate=0.0) for _ in range(num_layers)
        ])
        self.output_value_layer = nn.Linear(self.d_model, 1)
        self.output_probs_layer = nn.Linear(self.d_model, self.n_gens)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x = (x / self.color_devider).float()
        x = (x / self.color_devider).int() / 6.0

        x = self.input_linear(x)
        for layer in self.liear_blocks:
            x = layer(x)

        value = self.output_value_layer(x)
        probs = self.output_probs_layer(x)

        # print("x:", x.shape)

        return value, probs
    
class PilgrimCNN(nn.Module):
    def __init__(
        self,
        edge_size = 3,
        count_egdegs = 6,
        n_gens = 12,
        d_model = 16,
        num_layers = 4
    ):
        super(PilgrimCNN, self).__init__()
        self.space_size = edge_size * edge_size * count_egdegs
        self.edge_size = edge_size
        self.count_egdegs = count_egdegs
        
        self.n_gens = n_gens
        self.d_model = d_model
        self.num_layers = num_layers

        self.color_devider = int(self.space_size/count_egdegs) # 9

        self.input_cnn = CNNResidualBlock(in_channels=1, out_channels=self.d_model)
        self.cnn_blocks = nn.ModuleList([
            # nn.Linear(self.d_model, self.d_model) for _ in range(num_layers)
            CNNResidualBlock(self.d_model, self.d_model) for _ in range(num_layers)
        ])        

        self.output_value_layer = nn.Linear(864, 1)
        self.output_probs_layer = nn.Linear(864, self.n_gens)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (x / self.color_devider).int() / self.count_egdegs
        x = x.view(-1, self.edge_size*self.count_egdegs, self.edge_size).transpose(2, 1)
        x = x.unsqueeze(dim=1) 

        x = self.input_cnn(x)
        for layer in self.cnn_blocks:
            x = layer(x)

        x = x.flatten(start_dim=1)
        # print("x:", x.shape)

        value = self.output_value_layer(x)
        probs = self.output_probs_layer(x)
        
        # print("x:", x.shape)

        return value, probs

class PilgrimMLP2(nn.Module):
    def __init__(
            self,
            space_size = 54,
            n_gens = 12,
            d_model = 128,
            nhead = 4,
            num_layers = 4
        ):
        super(PilgrimMLP2, self).__init__()

        self.space_size = space_size
        self.n_gens = n_gens
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.color_devider = int(space_size/6) # 9

        self.input_embedding = nn.Embedding(
            num_embeddings=27, 
            embedding_dim=d_model
        )

        self.output_value_layer = nn.Linear(self.d_model, 1)

        # self.block_1_1 = FFBlock(hidden_in_dim=self.space_size * d_model, hidden_out_dim=self.d_model   )
        # self.block_1_2 = FFBlock(hidden_in_dim=self.d_model, hidden_out_dim=self.d_model * 2)
        # self.block_1_3 = FFBlock(hidden_in_dim=self.d_model * 2,hidden_out_dim=self.d_model * 1)

        self.blocks_1 = nn.ModuleList([
            FFBlock(self.space_size * d_model, hidden_out_dim=self.d_model),
            FFBlock(hidden_in_dim=self.d_model, hidden_out_dim=self.d_model * 2),
            FFBlock(hidden_in_dim=self.d_model * 2,hidden_out_dim=self.d_model * 1)
        ])

        self.blocks_2 = nn.ModuleList([
            FFBlock(self.space_size * d_model, hidden_out_dim=self.d_model),
            FFBlock(hidden_in_dim=self.d_model, hidden_out_dim=self.d_model * 2),
            FFBlock(hidden_in_dim=self.d_model * 2,hidden_out_dim=self.d_model * 1)
        ])        

        self.blocks_3 = nn.ModuleList([
            FFBlock(self.space_size * d_model, hidden_out_dim=self.d_model),
            FFBlock(hidden_in_dim=self.d_model, hidden_out_dim=self.d_model * 2),
            FFBlock(hidden_in_dim=self.d_model * 2,hidden_out_dim=self.d_model * 1)
        ])      

        self.blocks_4 = nn.ModuleList([
            FFBlock(self.space_size * d_model, hidden_out_dim=self.d_model),
            FFBlock(hidden_in_dim=self.d_model, hidden_out_dim=self.d_model * 2),
            FFBlock(hidden_in_dim=self.d_model * 2,hidden_out_dim=self.d_model * 1)
        ])          

        # self.block_2_1 = FFBlock(
        #     hidden_in_dim=self.space_size * d_model,
        #     hidden_out_dim=self.d_model   
        # )

        # self.block_2_2 = FFBlock(
        #     hidden_in_dim=self.d_model,
        #     hidden_out_dim=self.d_model * 2
        # )

        # self.block_2_3 = FFBlock(
        #     hidden_in_dim=self.d_model * 2,
        #     hidden_out_dim=self.d_model * 1
        # )

        # self.block_3_1 = FFBlock(
        #     hidden_in_dim=self.space_size * d_model,
        #     hidden_out_dim=self.d_model   
        # )

        # self.block_3_2 = FFBlock(
        #     hidden_in_dim=self.d_model,
        #     hidden_out_dim=self.d_model * 2
        # )

        # self.block_3_3 = FFBlock(
        #     hidden_in_dim=self.d_model * 2,
        #     hidden_out_dim=self.d_model * 1
        # )


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x = (x / self.color_devider).long()
        x = (x / 2.0).long()
        x = self.input_embedding(x)
        x = x.flatten(start_dim=1)

        # x_1 = (self.block_1_1(x) + self.block_1_2(x) + self.block_1_2(x)) / 3.0
        # x_2 = (self.block_2_1(x) + self.block_2_2(x) + self.block_2_3(x)) / 3.0

        # value = (self.output_value_layer(x_1) + self.output_value_layer(x_2)) / 2.0
        # policy = (self.output_probs_layer(x_1) + self.output_probs_layer(x_2)) / 2.0


        # x_1 = self.block_1_1(x)
        # x_1 = self.block_1_2(x_1)
        # x_1 = self.block_1_3(x_1)

        # x_2 = self.block_2_1(x)
        # x_2 = self.block_2_2(x_2)
        # x_2 = self.block_2_3(x_2)        

        # x_3 = self.block_3_1(x)
        # x_3 = self.block_3_2(x_3)
        # x_3 = self.block_3_3(x_3)        

        x_1 = x
        for layer in self.blocks_1:
            x_1 = layer(x_1)

        x_2 = x
        for layer in self.blocks_2:
            x_2 = layer(x_2)

        x_3 = x
        for layer in self.blocks_3:
            x_3 = layer(x_3)            

        x_4 = x
        for layer in self.blocks_4:
            x_4 = layer(x_4)            

        v_1 = self.output_value_layer(x_1)
        v_2 = self.output_value_layer(x_2)
        v_3 = self.output_value_layer(x_3)
        v_4 = self.output_value_layer(x_4)
        
        # print(f"v_1: {v_1[-1].item()}; v_2: {v_2[-1].item()}; v_3: {v_3[-1].item()}")
        # value = (v_1 + v_2 + v_3) / 3.0

        # vs = torch.cat([v_1, v_2, v_3], dim=1)
        # value, _ = torch.max(vs, dim=1)
        # value = value.unsqueeze(dim=1)

        vs = torch.cat([v_1, v_2, v_3, v_4], dim=1)
        value = torch.mean(vs, dim=1)
        value = value.unsqueeze(dim=1)

        # print(f"value: {value.shape}")
        # value = torch.mean(vs, dim=0)

        # print(f"value: {value.shape}")
        
        # policy = (
        #     self.output_probs_layer(x_1) + self.output_probs_layer(x_2) + self.output_probs_layer(x_3)
        # ) / 3.0

        return value, None

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_pilgrim():
    model = Pilgrim(
        input_dim = 54, 
        hidden_dim1 = 5000, 
        hidden_dim2 = 1000, 
        num_residual_blocks = 4
    ) # 14M params
    model.eval()
    print("No script")
    start = time.time()
    for _ in range(100):        
        with torch.no_grad():
            _ = model(torch.randint(low=0, high=54, size=(10240, 54)))
    end = time.time()
    duration = np.round(end - start, 3)
    print(f"Duration: {duration} sec")

    print("With script")
    model = model.to_script()
    model.eval()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(torch.randint(low=0, high=54, size=(10240, 54)))
    end = time.time()
    duration = np.round(end - start, 3)
    print(f"Duration: {duration} sec")

    print(model)
    print(np.round(count_parameters(model) / 1000), "K")


def check_pilgrim_transformer():
    model = PilgrimTransformer()
    print("Count params: ", int_to_human(count_parameters(model)))

    with torch.no_grad():
        value, policy = model(torch.randint(low=0, high=54, size=(1, 54)))

    print("Value:", value)
    print("Policy:", policy)

def check_pilgrim_simple():
    model = PilgrimSimple()
    print("Count params: ", int_to_human(count_parameters(model)))

    with torch.no_grad():
        value, policy = model(torch.randint(low=0, high=54, size=(1, 54)))

    print("Value:", value)
    print("Policy:", policy)

def check_pilgrim_cnn():
    model = PilgrimCNN()
    print("Count params: ", int_to_human(count_parameters(model)))

    with torch.no_grad():
        # value, policy = model(torch.randint(low=0, high=54, size=(1, 54)))
        value, policy = model(torch.arange(0, 54).int().unsqueeze(dim=0))

    print("Value:", value)
    print("Policy:", policy)

def check_pilgrim_mlp2():
    model = PilgrimMLP2()
    print("Count params: ", int_to_human(count_parameters(model)))

    with torch.no_grad():
        value, policy = model(torch.randint(low=0, high=54, size=(1, 54)))

    print("Value:", value)
    print("Policy:", policy)


if __name__ == "__main__":
    # check_pilgrim()
    # check_pilgrim_transformer()
    # check_pilgrim_simple()
    # check_pilgrim_cnn()
    check_pilgrim_mlp2()