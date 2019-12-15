#Policy Gradients with CartPole-v1
#Alps 
#12/13

import gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models, layers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout



max_episodes = 2048       #max trained episodes for the model
dropout_rate = 0.05
action_d = 2
state_d = 4
score_list = []  
model = Sequential()
model.add(Dense(64, input_dim=state_d, activation='relu'))
model.add(Activation('relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(action_d))
model.add(Activation('softmax'))
model.compile(loss = 'mse', optimizer=optimizers.Adam(0.001))




def cal_score(score):
        if(score > 400): return 1 + 1
        elif (score > 200): return 1 + 0.3        
        else: return 1

def act(s):
    prob = model.predict(np.array([s]))[0]
    return np.random.choice(len(prob), p=prob)

def discount_rewards(rewards, gamma=0.975):
    prior = 0
    out = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        prior = prior * gamma + rewards[i]
        out[i] = prior
    return out / np.std(out - np.mean(out))

def train(records):
    s_batch = np.array([record[0] for record in records])
    a_batch = np.array([[1 if record[1] == i else 0 for i in range(ACTION_DIM)]
                        for record in records])
    prob_batch = model.predict(s_batch) * a_batch
    r_batch = discount_rewards([record[2] for record in records])
    model.fit(s_batch, prob_batch, sample_weight=r_batch, verbose=0)


env = gym.make('CartPole-v1')
for i in range(max_episodes):
    s = env.reset()
    score = 0
    replay_records = []
    while True:
        a = act(s)
        next_s, r, done, _ = env.step(a)
        r = cal_score(score)
        replay_records.append((s, a, r))
        score += r
        s = next_s
        if done:
            train(replay_records)
            score_list.append(score)
            break
    
    if np.mean(score_list[-20:]) >= 500:
        model.save('CartPole-v0-pg.h5')
        break
env.close()


