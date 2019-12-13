#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import random
env = gym.make('CartPole-v0')
n=0
x=0
for i_episode in range(10):
    observation = env.reset()
    for step in range(100):
        env.render()
        #print(observation)
        dis=float(observation[0])
        v=float(observation[1])
        degree=float(observation[2])
        change=float(observation[3])
        if(degree>-0.1 and degree<0.1):
            action = random.randint(0,1)
            n=0
            if(abs(change)>0.8):
                if(v>=0):
                    action=0
                else:
                    action=1
        else:
            if(n==0):
                n=n+1
                last=change=float(observation[2])
                if(degree>0 and change>0):
                    if(v>=0):
                        action=0
                    else:
                        action=1
                elif(degree<0 and change<0):
                    if(v>=0):
                        action=0
                    else:
                        action=1
                else:
                    if(v>=0):
                        action=1
                    else:
                        action=0
                lastaction=action
            else:
                action=lastaction
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(step+1))
            break



