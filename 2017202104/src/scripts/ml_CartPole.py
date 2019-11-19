import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm, linear_model
from sklearn.neighbors import KNeighborsClassifier
import gym
import joblib

model_path = "C:/Users/bean/Desktop/train_model.m"
dataset_path = "C:/Users/bean/Desktop/CartPole_dataset.csv"

#using a specific model to decide next action
def take_action(model,observation):
    result = model.predict([observation])
    return int(result[0])

#using LogisticRegression
def train_model(target_model_path,model_type='logistic'):
    #data preprocessing
    data = pd.read_csv(dataset_path, header=-1)
    data_list = np.array(data).tolist()
    x_train = []
    y_train = []
    for i in range(len(data_list)):
        y_train.append(data_list[i][0])
        x_train.append(data_list[i][1:])
    x_train = preprocessing.scale(x_train)

    #train model
    if model_type=='logistic':
        model = linear_model.LogisticRegression(solver='liblinear')
        model.fit(x_train, y_train)
    elif model_type=='svm':
        model = svm.SVC(kernel='linear',C=1.0)
        model.fit(x_train, y_train)
    elif model_type=='knn':
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(x_train, y_train)
    else:
        print('invalid model name!')

    # save model
    joblib.dump(model, target_model_path)


def test(run_time=10):
    model = joblib.load(model_path)
    env = gym.make('CartPole-v0')
    for i_episode in range(run_time):
        observation = env.reset()
        for step in range(1000):
            env.render()
            action = take_action(model, observation)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(step + 1), " ", i_episode, "reward: ", reward)
                break
    env.close()

if __name__ == '__main__':
    train_model(model_path,'logistic')  #please train model when running the program at first time
    test(10)