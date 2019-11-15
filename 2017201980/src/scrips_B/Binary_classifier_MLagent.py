# -*- coding: utf-8 -*-
import gym
import pandas as pd
import matplotlib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import numpy as np

def load_csv_data(filename):     #读取文件 格式为每行（observation[0],.[1],.[2],.[3],action）
    data = []    
    labels = []    
    datafile = open(filename)    
    for line in datafile:    
        fields = line.strip().split(' ')    
        data.append([float(field) for field in fields[:-1]])    
        labels.append(fields[-1])    
    data = np.array(data)    
    labels = np.array(labels)    
    return data, labels    



#load data
X, y = load_csv_data('sample.out') 
#cross_exame
X_train,X_test,y_train,y_test = train_test_split(X,y)



### SVM Classifier 
print("==========================================")   
from sklearn.svm import SVC
clf1 = SVC(gamma='auto',kernel='rbf', probability=True)
clf1.fit(X_train,y_train)
predictions = clf1.predict(X_test)
print("SVM")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

### Logistic Regression Classifier!
print("==========================================")      
from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression(solver = "lbfgs",penalty='l2')
clf2.fit(X_train,y_train)
predictions = clf2.predict(X_test)
print("LR")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

### RandomForest!
print("==========================================")   
RF = RandomForestClassifier(n_estimators=10,random_state=11)
RF.fit(X_train,y_train)
predictions = RF.predict(X_test)
print("RF")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

### voting_classify
print("==========================================")   
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
#import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#clf1 = GradientBoostingClassifier(n_estimators=200)
clf2 = RandomForestClassifier(random_state=0, n_estimators=500)
clf3 = LogisticRegression(solver = "lbfgs",random_state=1)
# clf4 = GaussianNB()
#clf5 = xgboost.XGBClassifier()
clf = VotingClassifier(estimators=[
    #('gbdt',clf1),
    ('rf',RF),
     ('lr',clf2),
    # ('nb',clf4),
    # ('xgboost',clf5),
    ('SVM',clf1)
    ],
    voting='soft')
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("voting_classify")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

def nex_action(observation):
    result = RF.predict([observation])
    return int(result[0])


#simulation Demo
env=gym.make('CartPole-v1')      
for episode in range(10):
    observation = env.reset() 
    tmp = 0
    for t in range(500):
        #env.render() 
        action = nex_action(observation)
        observation, reward, done, info = env.step(action)  
        tmp += 1
        if done:
            print("Episode finished after {} timesteps".format(tmp))
            break        
    #print("reward: ", tmp)
env.close()