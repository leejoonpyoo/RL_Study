import numpy as np
import random

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4)) # (x, y, action) 3차원 배열
        self.eps = 0.9
        self.alpha = 0.01
        self.gamma = 0.1

    def select_action(self, s):
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0,3)
        else:
            action_value = self.q_table[x, y, :]
            action = np.argmax(action_value)
            return action
    
    def update_table(self, history):
        cum_reward = 0
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            x, y = s
            self.q_table[x, y, a] = self.q_table[x, y, a] + self.alpha * (cum_reward - self.q_table[x, y, a])
            cum_reward = r + self.gamma *cum_reward

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        # 학습이 각 위치에서 어느 행동을 하는 것이 좋을지를 보여주는 함수
        q_lst = self.q_table.tolist()
        data = np.zeros((5, 7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)
    
