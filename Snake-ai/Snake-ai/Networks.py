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
            
        self.CNN = nn.Sequential(
            nn.Conv2d(2,128,3,2,0,1,2),
            nn.SiLU(),
            nn.Conv2d(128,256,3,1,0,1,2),
            nn.SiLU(),
            nn.Conv2d(256,128,3,2),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(512,1280 ),
            nn.SiLU(),
            nn.Linear(1280,1280 ),
            nn.SiLU(),
            nn.Linear(1280,1280 ),
            nn.SiLU(),
            nn.Linear(1280,640 ),
            nn.SiLU(),
            nn.Linear(640,3 ),
            )
        '''
        self.FC_N = nn.Sequential(
            nn.Linear(9,1100 ),
            nn.SiLU(),
            nn.Linear( 1100,1100),
            nn.SiLU(),
            nn.Linear(1100, 1100),
            nn.SiLU(),
            nn.Linear( 1100,action_space)
            )
        '''
        
    def forward(self, matrixes,info):
        map_data = self.CNN(matrixes)
        return map_data
        '''
        map_data = map_data.squeeze()
        if np.shape(info) == torch.Size([1,3]) :
            merge = [map_data,info.squeeze()]
            map_data = torch.cat(merge)
        else :
            map_data = torch.cat((map_data,info.squeeze()),1)

        return self.FC_N(map_data)
        '''    

