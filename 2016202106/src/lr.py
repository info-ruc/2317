import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import random

def loss(y_predicted, y_target):
    return ((y_predicted - y_target)**2).sum()

def train(action):
    D = torch.tensor(pd.read_csv('data{}.csv'.format(action), header = None).values, dtype=torch.float)
    x_dataset = D[:,0:4].t()
    y_dataset = D[:,5:9].t()
    n= 4

    A = torch.randn((n, n), requires_grad = True)
    b = torch.randn((4, 1), requires_grad = True)

    optimizer = optim.Adam([A, b], lr = 0.001)

    for t in range(200000):
        optimizer.zero_grad()
        y_predicted = A.mm(x_dataset) + b
        current_loss = loss(y_predicted, y_dataset)
        current_loss.backward()
        optimizer.step()
        if current_loss < 0.001:
            return A.detach().numpy(), b.detach().numpy()

env = gym.make("CartPole-v1")
env._max_episode_steps = 20000
A, a = train(0)
B, b = train(1)
exact = []
predict = []
for i_episode in range(1):
    state = env.reset()
    exact.append(state)
    predict.append(state)
    print("=============================")
    for t in range(10000):
        env.render()
        y0 = np.matmul(A, state.reshape(4,1)) + a
        y1 = np.matmul(B, state.reshape(4,1)) + b
        if abs(y0[2]+y0[3]) > abs(y1[2] + y1[3]): 
            action = 1
            predict.append(y1.reshape(4))
        else:
            action = 0
            predict.append(y0.reshape(4))
        state, reward, done, info = env.step(action)
        exact.append(state)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

x0 = []
y0 = []
x1 = []
y1 = []
x2 = []
y2 = []
for j in range(min(750, len(exact))):
    x0.append(exact[j][0])
    y0.append(predict[j][0])
    x1.append(exact[j][1])
    y1.append(predict[j][1])
    x2.append(exact[j][2])
    y2.append(predict[j][2])
plt.figure(figsize = (18,7))
plt.plot(x0, 'g-+', label = 'exact_pos')
plt.plot(y0, 'r:*', label = 'predict_pos')
plt.legend()
plt.show()
plt.figure(figsize = (18,7))
plt.plot(x1, 'g-+', label = 'exact_v')
plt.plot(y1, 'r:*', label = 'predict_v')
plt.legend()
plt.show()
plt.figure(figsize = (18,7))
plt.plot(x2, 'g-+', label = 'exact_angel')
plt.plot(y2, 'r:*', label = 'predict_angel')
plt.legend()
plt.show()

env.close()
