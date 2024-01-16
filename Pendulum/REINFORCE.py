import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.input_layer = nn.Linear(state_dim, 512)
        self.mu_layer = nn.Linear(512, action_dim)
        self.log_std_layer = nn.Linear(512, action_dim)
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        
        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))
        
        return mu, log_std.exp()
    
class REINFORCE:
    def __init__(self, state_dim, action_dim, gamma=0.9):
        self.policy = Policy(state_dim, action_dim)
        self.gamma = gamma
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.0003)
        self.buffer = []
        
    @torch.no_grad()
    def act(self, s, training=True):
        self.policy.train(training)
        
        s = torch.as_tensor(s).float()
        mu, std = self.policy(s)
        z = torch.normal(mu, std) if training else mu
        a = torch.tanh(z)
        
        return a.numpy()
    
    def learn(self):
        # [(s_1, a_1, r_1), (s_2, a_2_r_2), ... ]를 (s_1, s_2, ...), (a_1, a_2, ...), (r_1, r_2, ...)로 변환
        s, a, r = map(np.stack, zip(*self.buffer))
        
        # G_t 만들어주기
        G = np.copy(r)
        for t in reversed(range(len(r) - 1)):
            G[t] += self.gamma * G[t + 1]
        s, a, G = map(lambda x: torch.as_tensor(x).float(), [s, a, G])
        G = G.unsqueeze(1)  # 열벡터 만들어주기
                
        # log prob 만들기
        mu, std = self.policy(s)
        m = torch.distributions.Normal(mu, std)
        z = torch.atanh(torch.clip(a, -1.0 + 1e-7, 1.0 - 1e-7))  # torch.atanh(-1.0), torch.atanh(1.0)은 각각 -infty, infty라서 clipping 필요
        log_prob = m.log_prob(z)
        
        # 손실함수 만들기 및 역전파
        policy_loss = - (log_prob * G).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
    def process(self, s, a, r, done):
        self.buffer.append((s, a, r))
        if done: 
            self.learn()
            self.buffer = []


