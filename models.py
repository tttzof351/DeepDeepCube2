import numpy as np
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
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