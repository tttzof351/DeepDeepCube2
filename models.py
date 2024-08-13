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
            num_layers=4
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


if __name__ == "__main__":
    # check_pilgrim()
    check_pilgrim_transformer()