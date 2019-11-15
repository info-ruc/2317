#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gym
weigth=np.ones(4)+np.random.rand(4)#[50.4,90.2,106.2,80]#
weight_best=np.ones(4)#[50.4,90.2,106.2,80]#
change_alpha=1
score_best=0
env = gym.make('CartPole-v1')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped
f=open("C:\\Users\\邵羽\\Desktop\\data.txt",'a')
for i_episode in range(20):
    weigth = weight_best
    observation = env.reset()
    print(observation)
    ob=observation
    score=0
    weigth += np.random.rand(4) * change_alpha
    while True:
        env.render()
        action = np.sum(np.multiply(weigth, observation))
        if action>0:
            action =1
        else:
            action =0
        observation_, reward, done, info = env.step(action)
        score+=reward
        f.write(str(action))
        f.write(" ")
        f.write(str(ob[0]))
        f.write(" ")
        f.write(str(ob[1]))
        f.write(" ")
        f.write(str(ob[2]))
        f.write(" ")
        f.write(str(ob[3]))
        f.write("\n")
        ob=observation_
        if done:
            if score > score_best:
                score_best = score
                weight_best = weigth
                change_alpha *= 0.5
            else:
                change_alpha *= 2
            break
        if score>1000 and np.sum(np.abs(weight_best-weigth))<0.001:
            print("best",score,"episode",i_episode,"w:",weight_best)
        observation = observation_
    if(score<1000):
        f.seek(0)
        f.truncate()
    else:
        break
env.close()
f.close()

