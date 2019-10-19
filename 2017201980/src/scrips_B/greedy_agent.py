#!/usr/bin/env python
# coding: utf-8


import gym
env = gym.make('CartPole-v1')
tmp = 0;

for i_episode in range(20):
    observation = env.reset()
    action = 1;
    for t in range(100):
        env.render()
        #print(observation)
        #action = env.action_space.sample()
        action = action ^ 1
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            tmp += t+1
            break
print("Average timesteps: {} ".format(tmp/20))
env.close()




