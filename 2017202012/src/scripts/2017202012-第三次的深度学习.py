# coding: utf8

import numpy as np
import tensorflow as tf
import gym

# 计算action潜在分数，r为当前action及之后每一步获得的reward数组
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):   #从后往前计算action累计得分
        running_add = running_add * decay + r[t]
        discounted_r[t] = running_add
    return discounted_r

# 创建环境
env = gym.make('CartPole-v0')

# 初始化环境
env.reset()
tf.reset_default_graph()

# 参数设置
d_obs = 4     #输入的环境信息维度
m = 50      # 隐藏层节点数
batch_size = 25        # 一次训练所选取的样本数
learning_rate = 0.1    #学习速率
decay = 0.99    #reward的衰减系数
eps_num = 1         #当前实验序号计数
eps_max = 10000    #总实验次数
reward_sum = 0              #累计reward
xs = []      # observation环境信息列表
ys = []      # label列表
rs = []     # 所记录每一个action的reward

#创建卷积层
observations = tf.placeholder(tf.float32, [None, d_obs], name='input_x')
w1 = tf.get_variable('w1', shape=[d_obs, m], initializer=tf.contrib.layers.xavier_initializer())    #权重1
layer1 = tf.nn.relu(tf.matmul(observations, w1))   #隐含层输出1：用relu激活函数处理obs乘w1
w2 = tf.get_variable('w2', shape=[m, 1], initializer=tf.contrib.layers.xavier_initializer())   #权重2
prob = tf.nn.sigmoid(tf.matmul(layer1, w2))   #隐含层输出2：用sigmoid激活函数处理输出1乘w2

# 模型用到的一些定义
disc_reward = tf.placeholder(tf.float32, name='reward_signal')  # 每个action的潜在分数
input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')  # 虚拟的label值（0/1）
prob_log = tf.log(input_y * (input_y-prob) + (1-input_y)*(input_y + prob))   # 当前action对应的概率的对数
loss = -tf.reduce_mean(prob_log * disc_reward)   # 损失函数 = 潜在分数 × 概率对数
var_train = tf.trainable_variables()   # 需要训练的变量
newGrads = tf.gradients(loss, var_train)    # 模型参数关于该损失函数的梯度

# 使用adam优化器
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
grad1 = tf.placeholder(tf.float32, name='batch_grad1')   #第一层神经网络的梯度
grad2 = tf.placeholder(tf.float32, name='batch_grad2')  #第二层神经网络的梯度
updateGrads = adam.apply_gradients(zip([grad1, grad2], var_train))  # 使用var_train中的参数计算梯度，并将计算结果更新至模型参数中

with tf.Session() as sess:
    ok_flag = 0                 #训练完成退出标志
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset()
    gradBuffer = sess.run(var_train)  #储存参数梯度的缓冲器
    for i, grad in enumerate(gradBuffer):  #初始化为0
        gradBuffer[i] = grad * 0
    print("average reward : ")

    while eps_num <= eps_max:   # 游戏最多能进行 eps_max次，每次循环是一次游戏中的一步
        x = np.reshape(observation, [1, d_obs])    # observation对应的策略网络输入格式
        # 将observation装入x中以输入策略网络，加入prob后run得概率tfprob：action取1的概率
        tfprob = sess.run(prob, feed_dict={observations:x})
        if np.random.uniform() < tfprob:  #按所得概率tfprob取action
            action = 1
        else:
            action = 0
        observation, reward, done, info = env.step(action)   #以action走一步
        reward_sum += reward
        
        # 将如下信息压入记录本轮游戏信息的列表
        xs.append(x)   #环境信息
        y = 1-action
        ys.append(y)   #label列表
        rs.append(reward)   #记录当前action的reward
        
        # 如果游戏结束
        if done:
            eps_num += 1    #游戏序号加1
            # 把这次游戏的各列表数据压入更大的矩阵列表 (列表中每个元素自成一行)
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(rs)
            #重置列表
            xs = []
            ys = []
            rs = []
            # 每个回合的潜在分数进行标准化：减均值后除以标准差
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            # 每回合更新一次模型参数，操作为newGrads
            tGrad = sess.run(newGrads, feed_dict={observations:epx, input_y:epy, disc_reward:discounted_epr})
            for i,grad in enumerate(tGrad): # 将每回合获得的梯度加到gradBuffer
                gradBuffer[i] += grad
        
            # 做了batch_size的整数倍次游戏时，完成了一次实验
            if eps_num % batch_size == 0:
                # 将gradBuffer里的参数更新到模型中
                sess.run(updateGrads, feed_dict={grad1:gradBuffer[0], grad2:gradBuffer[1]})
                # 参数缓冲器置零
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                reward_avg = reward_sum/batch_size
                batch_num = eps_num/batch_size
                print(' batch %d : %f'%(batch_num, reward_avg))
                # 当本次batch平均得分大于200时，结束程序。
                if reward_avg == 200:
                    print('Task solved in', eps_num, 'episodes (',int(batch_num),"batches )")
                    ok_flag += 1   # 完成训练退出标志
                    break
                # 重置batch对应的reward加和
                reward_sum = 0
            # 环境重置
            observation = env.reset()
            # 若模型训练完成（得分足够好）
            if ok_flag:
                break

#用训练出的模型做一次测试
print("---------finish training---------")
reward_sum = 0
    observation = env.reset()
    for i in range(205):
        env.render()
        x = np.reshape(observation, [1, d_obs])  # observation对应的策略网络输入格式 x
        tfprob = sess.run(prob, feed_dict={observations:x})  # action取1的概率
        if np.random.uniform() < tfprob:  #按所得概率tfprob取action
            action = 1
        else:
            action = 0
        observation, reward, done, info = env.step(action)   #以action走一步
        reward_sum += reward
        if done:      #若游戏未达满分而失败
            print("test reward: ", reward_sum)
            break
        if i == 199:  #若游戏一直进行到满分
            print("test reward: 200")

env.close()
