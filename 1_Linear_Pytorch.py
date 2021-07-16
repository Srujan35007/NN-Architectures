import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import transforms, datasets
print(f"Imports complete.")

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_block = nn.Sequential(
                nn.Linear(256,128),
                nn.ReLU(inplace=True),
                nn.Linear(128,64),
                nn.ReLU(inplace=True),
                nn.Linear(64,32),
                nn.ReLU(inplace=True),
                nn.Linear(32,16),
                nn.ReLU(inplace=True)
                )
    def __call__(self, x):
        return self.seq_block(x)

net = MyNet()
print(net)
