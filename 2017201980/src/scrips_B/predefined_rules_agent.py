#!/usr/bin/env python
# coding: utf-8


import gym
env = gym.make('CartPole-v1')
tmp = 0;

def next_move(observation,pre):
  return (observation[1] < -0.02 or(observation[1] <= 0 and pre == 0)) 
               
for i_episode in range(20):
    observation = env.reset()
    action = 0
    env.step(action)
    tmp = tmp + 1
    for t in range(100):
        env.render()
        #print(observation)
        #action = env.action_space.sample()
        #action = action ^ 1
        tt = action
        observation, reward, done, info = env.step(action)
        action = next_move(observation,tt)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            tmp += t+1
            break
print("Average timesteps: {} ".format((tmp+20)/20))
env.close()



