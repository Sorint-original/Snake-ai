from torch import nn
import torch



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
    def __init__(self,size) :
        super(SNAKE_Q_NET, self).__init__()
        layers = []
        self.neural_chunk = 80
        for i in range(1,size,1) :
            layers.append(nn.Conv2d(max(2,self.neural_chunk*(i-1)),self.neural_chunk*i,2,1,0,1,2))
            layers.append(nn.SiLU())
            
        layers.append(nn.Flatten(start_dim=0,end_dim=-1))
        self.CNN = nn.Sequential(*layers)
        self.FC_N = nn.Sequential(
            nn.Linear(self.neural_chunk*(size-1)+1, 1500),
            nn.SiLU(),
            nn.Linear( 1500,1500),
            nn.SiLU(),
            nn.Linear(1500, 1500),
            nn.SiLU(),
            nn.Linear( 1500,3)
            )

        
    def forward(self, matrixes,direction):
        map_data = self.CNN(matrixes)
        merge = [map_data,direction]
        map_data = torch.cat(merge)
        print(map_data)
        return self.FC_N(map_data)
