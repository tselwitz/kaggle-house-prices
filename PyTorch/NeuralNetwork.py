import torch
from torch import nn
  
class NeuralNetwork(nn.Module):
  def __init__(self, inp, outp):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(inp, 10),
      nn.ReLU(),
      nn.Linear(10, outp),
    )
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.layers(x)
