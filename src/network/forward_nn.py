import torch.nn as nn
import torch.nn.functional as F

class ForwardNN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(ForwardNN, self).__init__()
        self.layer_1 = nn.Linear(n_observations, 256)
        self.layer_2 = nn.Linear(256,256)
        self.layer_3 = nn.Linear(256, n_actions)
    

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)