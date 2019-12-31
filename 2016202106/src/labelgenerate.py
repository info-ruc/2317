import gym
import pandas as pd
import numpy as np
df = pd.DataFrame(np.zeros([1,5], float))
lastdf = df
env = gym.make('CartPole-v1')
env._max_episode_steps = 200000

def randomlabelgenerate(cnt, condition):
    while cnt > 0:
        observation = env.reset()
        for t in range(200000):
            env.render()
            action = env.action_space.sample()
            #if observation[2]+observation[3] < 0 or observation[1] > 1:
            #    action = 0
            #else:
            #    action = 1
            observation,reward,done,info=env.step(action)
            df.loc[0, 0:3] = observation
            df.loc[0, 4] = 1 if done else 0
            if done == condition:
                df.to_csv("label.csv", sep = ',', mode = 'a',header = False, index = False)
                cnt -= 1
            if done:
                break

def labelgenerate(cnt0 = 10000, cnt1 = 10000):
    while cnt0 > 0 or cnt1 > 0:
        observation = env.reset()
        states = []
        for t in range(20000):
            states.append(observation)
            env.render()
            if observation[0] + observation[1] > 2.4:
                action = 0
            elif observation[0] + observation[1] < -2.4:
                action = 1
            elif observation[2] + observation[3] > 0:
                action = 1
            else:
                action = 0
            observation,reward,done,info=env.step(action)
            
            if done:
                for i in range(int(0.9*t) - 1):
                    if cnt0 > 1:
                        df.loc[0, 0:3] = states[i]
                        df.loc[0, 4] = 0
                        df.to_csv("label.csv", sep = ',', mode = 'a',header = False, index = False)
                    cnt0 -= 1
                for i in range(int(0.9*t) - 1, len(states)):
                    if cnt1 > 1:
                        df.loc[0, 0:3] = states[i]
                        df.loc[0, 4] = 1
                        df.to_csv("label.csv", sep = ',', mode = 'a',header = False, index = False)
                    cnt1 -= 1
                
                break

labelgenerate(cnt0 = 10000, cnt1 = 10000)
env.close()

