import numpy as np
import gym
import time

def action(w, ob)
    wxb = np.dot(w[:4], ob) + w[4] 
    if wxb >= 0:
        return 1
    else:
        return 0

def getreward(env, w):
    ob= env.reset()
    sum = 0 
    for t in range(1000):
        act = action(w, ob) 
        ob, reward, done, info = env.step(act)
        sum+= reward
        if done:
            break
    return sum


def result(flag="random_guess"):
    env = gym.make("CartPole-v0")
    np.random.seed(10)
    tre= 0
    tw = np.random.rand(5) 

    for iter in range(10000):
        cw = None

        if flag == "hill_climbing":  
            cw = tw + np.random.normal(0, 0.1, 5)
        else: 
            cw = np.random.rand(5)


        reward= getreward(env, cw)
	
        if cur_sum_reward > best_reward:
            tre = reward
            tw = cw
	        if tre >= 200:
            break

    print(iter, tre, tw)
    return tre, tw
print(result("hill_climbing"))  
