#!/usr/bin/env python
# coding: utf-8

# In[4]:


import random
import gym
import numpy as np
from tensorflow.keras import models, layers

env = gym.make("CartPole-v0")  # 加载游戏环境

STATE_DIM, ACTION_DIM = 4, 2  # State 维度 4, Action 维度 2
model = models.Sequential([
    layers.Dense(64, input_dim=STATE_DIM, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(ACTION_DIM, activation='linear')
])
data_X=np.genfromtxt('C:\\Users\\邵羽\\Desktop\\l.csv',delimiter=',',usecols=(0,1,2,3))
data_Y=np.genfromtxt('C:\\Users\\邵羽\\Desktop\\l.csv',delimiter=',',usecols=(4,5))
model.compile(loss='mse', optimizer='adam')
i=1
for i in range(0,5):
    model.fit(data_X, data_Y)
import time
import gym
env = gym.make('CartPole-v0')
env.seed(1)     
env = env.unwrapped
for i_episode in range(10):
    observation = env.reset()
    score=0
    while True:
        env.render()
        action = np.argmax(model.predict(np.array([observation]))[0])
        observation_, reward, done, info = env.step(action)
        score+=reward
        observation=observation_
        if (score>5000):
            print("the score is more than 5000,spot")
            break
        if done:
            print("the score",score,"episode",i_episode)
            break
env.close()

