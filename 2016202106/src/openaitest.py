import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.DataFrame(np.zeros([1,9], float))
env = gym.make('CartPole-v0')
env._max_episode_steps = 20000
def run(method, ew):
    score = []
    for i_episode in range(20):
        observation = env.reset()
        print("=============================================")
        for t in range(10000):
            env.render()
            print(observation)
            #action = env.action_space.sample()
            #if observation[2]+observation[3] < 0 or observation[1] > 1:
             #   action = 0
            #else:
             #   action = 1
            if method == 'in_proportion':
                if abs(observation[0]) / 2.4 > abs(observation[2]) / np.cos(78 * np.pi / 180):
                    action = 0 if observation[0] > 0 else 1
                else:
                    action = 0 if observation[2] < 0 else 1
            elif method == 'angle':
                if observation[0] + observation[1] > 2.4:
                    action = 0
                elif observation[0] + observation[1] < -2.4:
                    action = 1
                elif observation[2] + observation[3] > 0:
                    action = 1
                else:
                    action = 0
            if ew:
                df.loc[0, 0:3] = observation
                df.loc[0, 4] = action
            observation,reward,done,info=env.step(action)
            if ew:
                df.loc[0, 5:8] = observation
                df.to_csv("data{}.csv".format(action), sep = ',', mode = 'a',header = False, index = False)
            if done:
                print(observation)
                print("Episode finished after {} timesteps".format(t+1))
                score.append(t+1)
                break
    return score

plt.figure(figsize=(16, 8))
plt.grid(True)
plt.plot(run('in_proportion',False), 'b:o', label = 'in proportion')
plt.plot(run('angle', False), 'r:v',label = 'angle')
plt.legend()
plt.show()
env.close()

