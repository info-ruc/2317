# coding: utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import  neighbors, linear_model
import gym
import time

def get_action(observation):
    result = logistic.predict([observation])
    return int(result[0])

#读取数据，划分数据
data = pd.read_csv("/Users/hewenyu/CartPoleData1.csv", header = -1)#读取数据
data_arr = np.array(data)#将dataframe转为数组
data_list=data_arr.tolist()#将数组转为list
x = []#存放预测所用属性
y = []#存放预测目标属性
for i in range(len(data_list)):#得到预测所用数据列表
    y.append(data_list[i][0])
    x.append(data_list[i][1:])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
x_scaled = preprocessing.scale(x_train)
scaler = preprocessing.StandardScaler().fit(x_train)
scaler.transform(x_test)

#训练逻辑回归分类模型
logistic = linear_model.LogisticRegression(solver='newton-cg')
logistic.fit(x_scaled, y_train)
#print('train score: ',logistic.score(x_scaled,y_train))
#print('test score: ',logistic.score(x_test,y_test))

#开始小车模型
env=gym.make('CartPole-v0')
observation = env.reset()       #初始化环境，observation为环境状态
count = 0
for t in range(1000):
    action = get_action(observation) #随机采样动作
    observation, reward, done, info = env.step(action)  #与环境交互，获得下一步的时刻
    count+=1
    if done:
        break
    env.render()         #绘制场景
    time.sleep(0.1)      #每次等待0.2s
print("reward: ", count)
env.close()
