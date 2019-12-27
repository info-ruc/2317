import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import random

def loss(y_predict, y_target):
    return ((y_predict - y_target)**2).sum()

def lineartrain(action):
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

class logisticmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x
def CrossEntropyLoss(y_predict, y_target):
    return torch.sum(-y_target*torch.log(y_predict) - (1 - y_target) * torch.log(1 - y_predict))

def logistictrain():
    model = logisticmodel()
    optimizer = optim.SGD(model.parameters(), lr = 0.00001)

    Data = torch.tensor(pd.read_csv('label.csv', header = None).values, dtype = torch.float)
    x_dataset = Data[:, 0:4].view(-1,4) #20000
    y_dataset = Data[:, 4].view(-1,1)
    i = 0
    lastloss = 0
    while True:
        optimizer.zero_grad()
        y_predict = model(x_dataset)
        loss = CrossEntropyLoss(y_predict, y_dataset)
        if (i % 10000 == 0):
            print(loss, '\n',x_dataset,'\n', y_dataset,'\n', model.linear.weight,'\n', y_predict)
        loss.backward()
        optimizer.step()
        if abs(lastloss - loss)< 0.0001:
            break
        lastloss = loss
    return model.linear.weight.detach().numpy()


theta = logistictrain()
theta = theta.reshape(1,4)

env = gym.make("CartPole-v0")
env._max_episode_steps = 20000
A, a = lineartrain(0)
B, b = lineartrain(1)

def run(method, ew):
    score = []
    for i_episode in range(10):
        observation = env.reset()
        print("=============================================")
        for t in range(10000):
            env.render()
            if method == 'in_proportion':
                if abs(observation[0]) / 2.4 > abs(observation[2]) / np.cos(78 * np.pi / 180):
                    action = 0 if observation[0] > 0 else 1
                else:
                    action = 0 if observation[2] < 0 else 1

            elif method == 'angle':
                if observation[0] + observation[1] > 2:
                    action = 0
                elif observation[0] + observation[1] < -2:
                    action = 1
                elif observation[2] + observation[3] > 0:
                    action = 1
                else:
                    action = 0
            elif method == 'lr':
                y0 = np.matmul(A, observation.reshape(4,1)) + a
                y1 = np.matmul(B, observation.reshape(4,1)) + b
                if abs(observation[2]) < 0.005:
                    action = 0 if observation[0] < 0 else 1
                elif abs(y0[2] + y0[3]) > abs(y1[2] + y1[3]):    # super 3000 average
                    action = 1
                else:
                    action = 0
            
            elif method == 'lrlogistic':
                y0 = np.matmul(A, observation.reshape(4,1)) + a
                y1 = np.matmul(B, observation.reshape(4,1)) + b
                if abs(observation[2]) < 0.01:  # < 2 angle
                    if np.matmul(theta, y0) > np.matmul(theta, y1):
                        action = 0
                    else:
                        action = 1
                elif abs(y0[2] + y0[3]) > abs(y1[2] + y1[3]):    # super 3000 average
                    action = 1
                else:
                    action = 0
            if ew:
                df.loc[0, 0:3] = observation
                df.loc[0, 4] = action
            observation,reward,done,info=env.step(action)
            if ew:
                df.loc[0, 5:8] = observation
                df.to_csv("data{}.csv".format(action), sep = ',', mode = 'a',header = False, index = False)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                score.append(t+1)
                break
    return score

plt.figure(figsize=(16, 8))
plt.grid(True)
plt.plot(run('in_proportion',False), 'b:o', label = 'in proportion')
#plt.plot(run('angle', False), 'r:v',label = 'angle')
plt.plot(run('lr', False), 'c:+',label = 'lr')
plt.plot(run('lrlogistic', False), 'k:d',label = 'lrlogistic')
plt.legend()
plt.show()

