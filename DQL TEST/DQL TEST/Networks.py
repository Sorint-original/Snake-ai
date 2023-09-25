import torch.nn  as nn
import torch.nn.functional as F


class Model1(nn.Module):
    def __init__(self,_state_size,_action_size) :
        super(Model1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(_state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,_action_size)
            )
        
    def forward(self, x):
        return self.model(x)




