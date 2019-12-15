import time
import warnings

import gym
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from machine_learning_agent import ClassicMachineLearningAgent

ENV_NAME = 'CartPole-v1'
warnings.filterwarnings('ignore')


def get_train_data():
    X = []
    y = []
    agent = ClassicMachineLearningAgent()
    agent.train()
    env = gym.make(ENV_NAME)
    for i in range(30):
        obs = env.reset()
        sum_reward = 0.0
        while True:
            action = agent.get_action(obs)
            X.append(obs[:])
            y.append(action)
            obs, reward, done, info = env.step(action)
            sum_reward += reward
            if done:
                # print(sum_reward)
                break
    env.close()
    print("train_data_ready")
    return np.array(X), np.array(y)


class DeepLearningAgent:
    """
    Author: 李昕旸
    """

    def __init__(self):
        tf.reset_default_graph()
        self.X_data, self.y_data = get_train_data()
        # 定义训练数据的占位符， x是特征值， y是标签值(0 或 1)
        self.x = tf.placeholder(tf.float32, [1, 4], name="X")
        self.y = tf.placeholder(tf.float32, [1, 1], name="Y")

        with tf.name_scope("Model1"):
            def model(x):
                d1 = tf.nn.relu(tf.layers.dense(x, 128))
                d2 = tf.nn.relu(tf.layers.dense(d1, 32))
                d3 = tf.layers.dense(d2, 1)
                return d3

            self.pred = model(self.x)

        self.train_epochs = 25  # 迭代次数
        self.learning_rate = 0.001  # 学习率

        # 定义损失函数 采用交叉熵作为损失函数
        with tf.name_scope("LossFunction"):
            self.loss_function = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.pred)

        # 使用自适应矩估计优化器 设置学习率和优化目标损失最小化
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_function)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self):

        loss_list = []  # 用于保存loss的值
        # 迭代训练
        for epoch in range(self.train_epochs):
            loss_sum = 0.0
            for xs, ys in zip(self.X_data, self.y_data):
                xs = xs.reshape(1, 4)
                ys = ys.reshape(1, 1)

                _, loss = self.sess.run([self.optimizer, self.loss_function], feed_dict={self.x: xs, self.y: ys})

                loss_sum = loss_sum + loss  # 累加损失

            X_train, y_train = shuffle(self.X_data, self.y_data)  # 打乱数据顺序 避免过拟合假性学习

            loss_average = loss_sum / len(y_train)  # 所有数据的平均损失
            loss_list.append(loss_average)

        # print(loss_list)

    def car_test(self):

        env = gym.make(ENV_NAME)
        for i in range(3):
            obs = env.reset()
            sum_reward = 0.0
            while True:
                time.sleep(0.01)
                env.render()
                obs_xs = np.array(obs).reshape(1, 4)
                choice_tmp = self.sess.run([self.pred], feed_dict={self.x: obs_xs})
                real_choice = 0 if choice_tmp[0] < 0.5 else 1
                obs, reward_tmp, done, info = env.step(real_choice)
                sum_reward += reward_tmp
                if done:
                    print(sum_reward)
                    break


if __name__ == '__main__':
    deep_learning_agent = DeepLearningAgent()
    deep_learning_agent.train()
    deep_learning_agent.car_test()
