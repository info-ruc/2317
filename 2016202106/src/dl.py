import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

MEMORY_SIZE = 10000
EPISODES = 100
GAMMA = 0.99
LR = 0.001
HIDDEN_LAYER = 128
BATCH_SIZE = 256

class Network(nn.Module):
    def __init__(self, hidden_layer):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, hidden_layer)
        self.l2 = nn.Linear(hidden_layer, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class Agent:
    def __init__(self, capacity = MEMORY_SIZE, gamma = GAMMA, lr = LR, hidden_layer = HIDDEN_LAYER, batch_size = BATCH_SIZE):
        self.gamma = gamma
        self.lr = lr
        self.capacity = capacity
        self.batch_size = BATCH_SIZE
        self.enet = Network(hidden_layer)
        self.optimizer = optim.Adam(self.enet.parameters(), lr = self.lr)
        self.memory = []
        self.cnt = 0

    def push(self, transition):
        if self.cnt >= self.capacity:
            self.memory[self.cnt % self.capacity] = transition
        else:
            self.memory.append(transition)
        self.cnt += 1
    
    def egreedy_action(self, state, timestep):
        epsilon = 1 / math.sqrt(timestep + 1)
        if random.random() < epsilon:
            return random.choice([0, 1])
        else:
            state = torch.tensor(state, dtype = torch.float).view(1, -1)
            action  = torch.argmax(self.enet(state)).item()
            return action

    def learn(self):
        if(self.cnt < self.batch_size):
            return
        
        samples = random.sample(self.memory, self.batch_size)
        state, action, next_state, reward = zip(*samples)

        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long).view(self.batch_size, -1)
        next_state = torch.tensor(next_state, dtype = torch.float)
        reward = torch.tensor(reward, dtype = torch.float).view(self.batch_size, -1)

        y_predict = self.enet(state).gather(1, action)
        y_target = reward + self.gamma * torch.max(self.enet(next_state).detach(), dim=1)[0].view(self.batch_size, -1)
        #print("y_predict:\n",y_predict)
        #print("y_target:\n", y_target)
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_predict, y_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


env = gym.make('CartPole-v1')

agent = Agent()

for episode in range(EPISODES):
    state = env.reset()
    for i in range(10000):
        env.render()
        action = agent.egreedy_action(state, i)
        #print(action)
        next_state, reward, done, info = env.step(action)
        if done:
            reward = -100

        agent.push([state, action, next_state, reward])
        state = next_state
        agent.learn()
        if done:
            print("finished after {} timesteps".format(i+1))
            break

env.close()
