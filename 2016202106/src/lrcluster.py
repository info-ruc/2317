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

def bfs():
    i = 0
    l = []

    l.append([0.0, 0.0, 0.0, 0.0, 0, 0])
    while i < len(l):
        x = np.array(l[i])
        #print(x, x[0:4], x[0:4].reshape(4, 1))
        u1 = np.matmul(A, x[0:4].reshape(4, 1)) + a
        u2 = np.matmul(B, x[0:4].reshape(4, 1)) + b
        #print(u1)
        #print(u1[0][0])
        if abs(u1[0]) >= abs(x[0]) and abs(u1[0]) < 2.4 and abs(u1[2]) < 0.5:
            l.append([u1[0][0], u1[1][0], u1[2][0], u1[3][0], x[4]+1, 0])
        if abs(u2[0]) >= abs(x[0]) and abs(u2[0]) < 2.4 and abs(u2[2]) < 0.5:
            l.append([u2[0][0], u2[1][0], u2[2][0], u2[3][0], x[4]+1, 1])
        i += 1
    df = pd.DataFrame(l)
    df.to_csv('cluster.csv', sep = ',', mode = 'w', header = False, index = None)

class Kmeans():
    def __init__(self, n_clusters = 18):
        self.n_clusters = n_clusters
        self.labels = None
        self.centroid = None
        self.level = None
    
    def fit(self, x):
        x = np.array(x)
        #init_row = np.array([2**i for i in range(int(np.log2(len(x)) - 1))])
        
        self.labels = np.zeros(len(x))
        self.centroid = np.array(x[init_row])
        self.level = np.zeros(self.n_clusters)
        while True:
            shift = np.zeros(6)
            for i, sample in enumerate(x):
                dist = ((sample[0:4]-self.centroid[:,0:4])**2).sum(axis = 1)
                self.labels[i] = np.argmin(dist)
            for j in range(self.n_clusters):
                new_centroid = np.mean(x[self.labels == j], axis = 0)
                shift += abs(new_centroid - self.centroid[j])
                self.centroid[j] = new_centroid
            print('shift', shift)
            print(shift.sum() < 1)
            if shift.sum() < 1000:
                break
        print(self.centroid)
        for j in range(self.n_clusters):
            self.level[j] = self.centroid[j][4]
    
    def predict(self, state):
        state = np.array(state).reshape(4)
        dist = ((state - self.centroid[:,0:4])**2).sum(axis = 1)
        return float(self.centroid[np.argmin(dist)][4])



env = gym.make("CartPole-v1")
env._max_episode_steps = 20000
A, a = lineartrain(0)
B, b = lineartrain(1)
bfs()
quit()
kmeans = Kmeans()
y = np.array(pd.read_csv('cluster.csv', header = None).values)
kmeans.fit(y)
timesteps = []
for i_episode in range(5):
    state = env.reset()
    print("=============================")
    score = []
    for t in range(10000):
        env.render()
        y0 = np.matmul(A, state.reshape(4,1)) + a
        y1 = np.matmul(B, state.reshape(4,1)) + b
        
        #if abs(y0[2] + y0[3]) + abs(y0[1]) + abs(y0[0])> abs(y1[2] + y1[3]) + abs(y1[1]) + abs(y1[0]):    # super 3000 average
        #print(kmeans.level(y0))

        print(kmeans.predict([1,1,1,1]))
        if kmeans.predict(y0.reshape(4)) > kmeans.predict(y1.reshape(4)):
            action = 1
        else:
            action = 0
        state, reward, done, info = env.step(action)

        if done:
            timesteps.append(t+1)
            break
env.close()
plt.figure(figsize = (16, 8))
plt.grid(True)
plt.plot(timesteps, 'g-', label = 'timesteps')
plt.legend()
plt.axis('tight')
plt.show()

