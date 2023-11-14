from torch import nn
import torch
import numpy as np
import time



class Q_net(nn.Module):
    def __init__(self,_state_size,_action_size) :
        super(Q_net, self).__init__()
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
    

class SNAKE_Q_NET(nn.Module):
    def __init__(self,size,action_space) :
        super(SNAKE_Q_NET, self).__init__()
        layers = []
        self.neural_chunk = 80
        for i in range(1,size-1,1) :
            layers.append(nn.Conv2d(max(2,self.neural_chunk*(i-1)),self.neural_chunk*i,2,1,0,1,2))
            layers.append(nn.SiLU())

        layers.append(nn.Conv2d(self.neural_chunk*(size-2),6,2,1,0,1,2))
            
        self.CNN = nn.Sequential(*layers)
        self.FC_N = nn.Sequential(
            nn.Linear(9,1100 ),
            nn.SiLU(),
            nn.Linear( 1100,1100),
            nn.SiLU(),
            nn.Linear(1100, 1100),
            nn.SiLU(),
            nn.Linear( 1100,action_space)
            )

        
    def forward(self, matrixes,info):
        map_data = self.CNN(matrixes)
        map_data = map_data.squeeze()
        if np.shape(info) == torch.Size([1,3]) :
            merge = [map_data,info.squeeze()]
            map_data = torch.cat(merge)
        else :
            map_data = torch.cat((map_data,info.squeeze()),1)

        return self.FC_N(map_data)

