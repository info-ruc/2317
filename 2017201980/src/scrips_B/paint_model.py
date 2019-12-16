#Policy Gradients with CartPole-v1
#Alps 
#12/13

import gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models, layers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

plt.plot(score_list)
x = np.array(range(len(score_list)))
smooth_func = np.poly1d(np.polyfit(x, score_list, 3))
plt.plot(x, smooth_func(x), label='Mean', marker='o', linestyle=':',color = 'black',linewidth=1, markersize=4)
print(plt.style.available)
plt.style.use('ggplot')
plt.show()
