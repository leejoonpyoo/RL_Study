import gym
import torch
from torch.distributions import Categorical
from src.PPO import PPO
from src.REINFORCE import REINFORCE
from src.ActorCritic import ActorCritic
import wandb

#Hyperparameters
learning_rate = 0.0005
gamma        = 0.98
lmbda        = 0.95
eps_clip     = 0.1  # epsilon for clipping
k_epoch      = 3    # update policy for K epochs, 동일한 데이터 배치를 이용해 K번 업데이트
T_horizon    = 20   # update policy every T timesteps

env_name = 'CartPole-v1'
#agent_name = 'PPO'
#agent_name = 'REINFORCE'
agent_name = 'ActorCritic'

        
def main(): 
    for seed in range(10):
        wandb.init(project=env_name, group=agent_name, name=f'test {seed+1}')

        env = gym.make(env_name, render_mode='human')

        if isinstance(env.observation_space, gym.spaces.Box):
            state_dim = env.observation_space.shape[0]
        else:
            state_dim = env.observation_space.n

        if isinstance(env.action_space, gym.spaces.Box):
            action_dim = env.action_space.shape[0]
        else:
            action_dim = env.action_space.n  

        #model = PPO(state_dim, action_dim)
        #model = REINFORCE(state_dim, action_dim)
        model = ActorCritic(state_dim, action_dim)
        score = 0.0
        print_interval = 20

        for n_epi in range(500):
            s, _ = env.reset()
            done = False
            
            if agent_name == 'PPO' or agent_name == 'ActorCritic':
                while not done:
                    for t in range(T_horizon):
                        prob = model.pi(torch.from_numpy(s).float())
                        m = Categorical(prob)
                        a = m.sample().item()
                        s_prime, r, done, truncated, info = env.step(a)

                        model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                        
                        s = s_prime
                        score += r
                        if done:
                            break

                    model.train_net()
                    
                    if score >= 50000:
                        done = True
            
            if agent_name == 'REINFORCE':
                while not done: # CartPole-v1 forced to terminates at 500 step.
                    prob = model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    s_prime, r, done, truncated, info = env.step(a)
                    model.put_data((r,prob[a]))
                    s = s_prime
                    score += r

                    if score >= 50000:
                        done = True
                    
                model.train_net()  

            if n_epi%print_interval==0 and n_epi!=0:
                print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
                wandb.log({'Steps': n_epi, 'AvgEpRet': score/print_interval})
                score = 0.0
        wandb.finish()
        env.close()

if __name__ == '__main__':
    main()