#Alps_12/13
import time
import numpy as np
import gym
from tensorflow.keras import models


saved_model = models.load_model('CartPole_v1')
env = gym.make("CartPole-v1")

for i in range(5):
    s = env.reset()
    score = 0
    while True:
        time.sleep(0.01)
        env.render()
        prob = saved_model.predict(np.array([s]))[0]
        a = np.random.choice(len(prob), p=prob)
        s, r, done, _ = env.step(a)
        score += r
        if done:
            print('using policy gradient, score: ', score)              break
env.close()
