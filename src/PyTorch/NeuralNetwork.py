import torch
from torch import nn
  
class NeuralNetwork(nn.Module):
  def __init__(self, inp, outp, hf):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(inp, hf),
      nn.ReLU(),
      nn.ReLU(),
      nn.Linear(hf, outp),
    )
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.layers(x)
