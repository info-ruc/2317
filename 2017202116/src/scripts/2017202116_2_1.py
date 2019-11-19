#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
from sklearn import svm
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics 
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn import ensemble
data=numpy.genfromtxt('C:\\Users\\邵羽\\Desktop\\s.csv',delimiter=',',usecols=(1,2,3,4))
target=numpy.genfromtxt('C:\\Users\\邵羽\\Desktop\\s.csv',delimiter=',',usecols=(0))
clf = GaussianNB()
train, test, t_train, t_test = train_test_split(data, target, test_size=0.60, random_state=0)
clf.fit(train, t_train)
Ada1 = AdaBoostClassifier( GaussianNB(),
                         n_estimators=100,algorithm="SAMME", learning_rate=0.2)
Ada1.fit(train, t_train)   
print("使用naive_bayes分类器")
print("准确率;",clf.score(test,t_test))
import gym
env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped
for i_episode in range(10):
    observation = env.reset()
    score=0
    while True:
        env.render()
        action=int(clf.predict([observation])[0])
        observation_, reward, done, info = env.step(action)
        score+=reward
        observation=observation_
        if (score>5000):
            print("the score is more than 5000,spot")
            break
        if done:
            print("the score",score,"episode",i_episode)
            break
print("使用svm分类器")
clf_rbf=svm.SVC(C=0.8, kernel="rbf",gamma=20,decision_function_shape='ovr')
clf_rbf.fit(train, t_train)
Ada2 = AdaBoostClassifier( svm.SVC(C=0.8, kernel="rbf",gamma=20,decision_function_shape='ovr'),
                         n_estimators=300,algorithm="SAMME",learning_rate=0.2)
Ada2.fit(train, t_train)
print("准确率;",clf_rbf.score(test,t_test))
for i_episode1 in range(10):
    observation = env.reset()
    score=0
    while True:
        env.render()
        action=int(clf_rbf.predict([observation])[0])
        observation_, reward, done, info = env.step(action)
        score+=reward
        observation=observation_
        if (score>5000):
            print("the score is more than 5000,spot")
            break
        if done:
            print("the score",score,"episode",i_episode1)
            break
print("使用naive_bayes分类器的集成学习 AdaBoostClassifier模型")
print("准确率;",Ada1.score(test,t_test))
for i_episode2 in range(10):
    observation = env.reset()
    score=0
    while True:
        env.render()
        action=int(Ada1.predict([observation])[0])
        observation_, reward, done, info = env.step(action)
        score+=reward
        observation=observation_
        if (score>5000):
            print("the score is more than 5000,spot")
            break
        if done:
            print("the score",score,"episode",i_episode2)
            break
print("使用svm分类器的集成学习 AdaBoostClassifier模型")
print("准确率;",Ada2.score(test,t_test))
for i_episode3 in range(10):
    observation = env.reset()
    score=0
    while True:
        env.render()
        action=int(Ada2.predict([observation])[0])
        observation_, reward, done, info = env.step(action)
        score+=reward
        observation=observation_
        if (score>5000):
            print("the score is more than 5000,spot")
            break
        if done:
            print("the score",score,"episode",i_episode3)
            break
params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 3, 'learning_rate': 0.01, 'loss': 'ls'}
clfs = ensemble.GradientBoostingRegressor(**params)
clfs.fit(train, t_train)
print("使用集成学习 AdaBoostRegressor模型")
print("准确率;",clfs.score(test,t_test))
for i_episode4 in range(10):
    observation = env.reset()
    score=0
    while True:
        env.render()
        action=int(clfs.predict([observation])[0])
        observation_, reward, done, info = env.step(action)
        score+=reward
        observation=observation_
        if (score>5000):
            print("the score is more than 5000,spot")
            break
        if done:
            print("the score",score,"episode",i_episode4)
            break
env.close()

