#!/usr/bin/env python
# coding: utf-8


import gym
import time
import numpy as np
tmp = 0
env = gym.make('CartPole-v1')

f = open("sample.out","w")

#arr = [0.32455586, -0.09436489,  1.42703162,  1.14888277, -0.0177973]
#arr = [0.19566202, 0.11578184, 0.7173747,  1.48423667, 0.05098461]
#arr = [ 1.92704091e-01,  3.80987661e-01,  1.32745303e+00,  2.07162982e+00, -9.27898585e-04]
arr = [0.01159834, 0.26770383, 1.31941917, 1.93764616, 0.00291291]
def next_move(observation):
    if observation.dot(arr[0:4]) + arr[4] > 0:
        return 1
    else :
        return 0
    
    
for i_episode in range(20):
    observation = env.reset()
    action = 0
    #env.step(action)
    for t in range(501):
        #env.render()
        observation, reward, done, info = env.step(action)
        action = next_move(observation)
        f.write(str(observation[0]) + " " + str(observation[1]) + " " + str(observation[2]) + " " + str(observation[3]) + " " + str(action) + "\n")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            tmp += t+1
            break
print("Average timesteps: {} ".format((tmp)/10))
env.close()
f.close()