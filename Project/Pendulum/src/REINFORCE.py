import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

#Hyperparameters
learning_rate  = 0.0003
gamma           = 0.9

class REINFORCE(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(REINFORCE, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(state_dim,128)
        self.fc_mu = nn.Linear(128,action_dim)
        self.fc_std  = nn.Linear(128,action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #self.optimization_step = 0

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        mu = 2.0*torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
    def put_data(self, item):
        self.data.append(item)      
    
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []
