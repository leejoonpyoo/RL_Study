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
    
    # 이 부분에서 MC와 차이가 있다.
    def update_table(self, transition):
        s,a,r,s_prime = transition
        x, y = s
        next_x, next_y = s_prime
        a_prime = self.select_action(s_prime)
        self.q_table[x, y, a] = self.q_table[x, y, a] + self.alpha * (r + self.gamma * self.q_table[next_x, next_y, a_prime] - self.q_table[x, y, a])

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        q_lst = self.q_table.tolist()
        data = np.zeros((5, 7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)
    
