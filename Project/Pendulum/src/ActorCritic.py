import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

#Hyperparameters
learning_rate  = 0.0003
gamma           = 0.9
lmbda           = 0.9
T_horizon       = 20 

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(state_dim,128)
        self.fc_mu = nn.Linear(128,action_dim)
        self.fc_std  = nn.Linear(128,action_dim)
        self.fc_v = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #self.optimization_step = 0

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        mu = 2.0*torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
    
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done_mask
        delta = td_target - self.v(s)

        mu, std = self.pi(s, softmax_dim=1)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(a)
        loss = -torch.log(log_prob) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
        
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()