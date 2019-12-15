#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import gym
import numpy as np
from tensorflow.keras import models, layers, optimizers

env = gym.make('CartPole-v0')

STATE_DIM, ACTION_DIM = 4, 2
model = models.Sequential([
    layers.Dense(100, input_dim=STATE_DIM, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(ACTION_DIM, activation="softmax")
])
model.compile(loss='mean_squared_error',
              optimizer=optimizers.Adam(0.001))


def choose_action(s):
    """预测动作"""
    prob = model.predict(np.array([s]))[0]
    return np.random.choice(len(prob), p=prob)


def discount_rewards(rewards, gamma=0.95):
    """计算衰减reward的累加期望，并中心化和标准化处理"""
    prior = 0
    out = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        prior = prior * gamma + rewards[i]
        out[i] = prior
    return out / np.std(out - np.mean(out))


def train(records):
    s_batch = np.array([record[0] for record in records])
    # action 独热编码处理，方便求动作概率，即 prob_batch
    a_batch = np.array([[1 if record[1] == i else 0 for i in range(ACTION_DIM)]
                        for record in records])
    # 假设predict的概率是 [0.3, 0.7]，选择的动作是 [0, 1]
    # 则动作[0, 1]的概率等于 [0, 0.7] = [0.3, 0.7] * [0, 1]
    prob_batch = model.predict(s_batch) * a_batch
    r_batch = discount_rewards([record[2] for record in records])
    model.fit(s_batch, prob_batch, sample_weight=r_batch, verbose=0)


episodes = 2000  # 至多2000次
score_list = []  # 记录所有分数
for i in range(episodes):
    s = env.reset()
    score = 0
    replay_records = []
    while True:
        a = choose_action(s)
        next_s, r, done, _ = env.step(a)
        replay_records.append((s, a, r))

        score += r
        s = next_s
        if done:
            train(replay_records)
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break
    # 最后10次的平均分大于 195 时，停止并保存模型
    if np.mean(score_list[-10:]) > 195:
        break
env.close()
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
        prob = model.predict(np.array([ observation]))[0]
        action = np.random.choice(len(prob), p=prob)
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

