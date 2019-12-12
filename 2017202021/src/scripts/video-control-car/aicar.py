import cv2
import pygame
import sys
import serial
import torch.nn as nn
import torch
import torch.optim as optim
import os
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import time


#采集图像数据
def get_data():
    pygame.init()  # 初始化pygame
    pygame.key.set_repeat(100, 100)#设置键盘连续输入
    size = width, height = 320, 240  # 设置窗口大小
    screen = pygame.display.set_mode(size)  # 显示窗口
    cap = cv2.VideoCapture(1)#读入图像
    bt = serial.Serial('COM8', 9600)#设置小车蓝牙端口
    img_path = 'photo/'
    i = 0
    while True:  # 死循环确保窗口一直显示
        for event in pygame.event.get():  # 遍历所有事件
            keys_pressed = pygame.key.get_pressed()
            #根据不同的方位存放不同的图片信息
            if keys_pressed[pygame.K_RIGHT]:
                print('向右移动')
                ret,img = cap.read()
                img = cv2.resize(img,(200,200))
                bt.write(4)#控制小车运行
                s = img_path + str(4) + '-image-' + str(i) + '.png'
                if ret == True:
                    cv2.imwrite(s,img)#存储图像
                i = i + 1
            elif keys_pressed[pygame.K_LEFT]:
                print('向左移动')
                ret, img = cap.read()
                img = cv2.resize(img, (200, 200))
                bt.write(3)  # 控制小车运行
                s = img_path + str(3) + '-image-' + str(i) + '.png'
                if ret == True:
                    cv2.imwrite(s, img)
            elif keys_pressed[pygame.K_UP]:
                print('向前移动')
                ret, img = cap.read()
                img = cv2.resize(img, (200, 200))
                bt.write(1)  # 控制小车运行
                s = img_path + str(1) + '-image-' + str(i) + '.png'
                if ret == True:
                    cv2.imwrite(s, img)
                i = i + 1
            elif keys_pressed[pygame.K_DOWN]:
                print('向后移动')
                ret, img = cap.read()
                img = cv2.resize(img, (200, 200))
                bt.write(2)  # 控制小车运行
                s = img_path + str(2) + '-image-' + str(i) + '.png'
                if ret == True:
                    cv2.imwrite(s, img)
                i = i + 1
            elif keys_pressed[pygame.K_SPACE]:
                print('停止')
                ret, img = cap.read()
                img = cv2.resize(img, (200, 200))
                bt.write(0)  # 控制小车运行
                s = img_path + str(0) + '-image-' + str(i) + '.png'
                if ret == True:
                    cv2.imwrite(s, img)
                i = i + 1
            if event.type == pygame.QUIT:  # 如果单击关闭窗口，则退出
                sys.exit()
    cap.release()
    s.close()
    return


#将采集的数据读入内存中
def load_data():
    data = []
    label = []
    filelist = os.listdir('photo/')
    one_hot = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]#用独热码代表停前后左右五个状态
    for file in filelist:
        dir = 'photo/' + file
        img = cv2.imread(dir)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,img_fixed = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
        #读取图像并进行处理
        a = file.split('-')
        L = int(a[0])
        label.append(np.array(one_hot[L]))
        b = np.array(img_fixed)
        b = b.reshape(-1)
        b = b + 1
        data.append(b)
    return data,label


#训练模型
#自定义定义多层感知机
class MLP(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size,output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def train():
    data, label = load_data()
    input_size = data[0].shape[0]
    hidden1_size = 1000
    hidden2_size = 100
    output_size = 5
    epoch = 10  # 训练的次数
    learning_rate = 1e-4  # 学习率
    mlp = MLP(input_size,hidden1_size,hidden2_size,output_size)#初始化模型

    loss_fn = nn.MSELoss(reduction='sum')#损失函数
    optimizer = optim.SGD(mlp.parameters(),lr=learning_rate)#优化方法

    mlp.train()
    for t in range(epoch):
        for i in range(len(data)):
            im = Variable(torch.from_numpy(data[i]).float())
            L = Variable(torch.from_numpy(label[i]).float())
            pred = mlp(im)
            loss = loss_fn(pred, L)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #训练模型
    mlp.eval()
    for i in range(len(data)):
        im = Variable(torch.from_numpy(data[i]).float())
        L = Variable(torch.from_numpy(label[i]).float())
        pred = mlp(im)
        print(pred,L)
    return mlp

#通过传入的图像进行预测并给出正确的运动方向
def pre():
    mlp = train()
    mlp.eval()
    cap = cv2.VideoCapture(1)  # 读入图像
    bt = serial.Serial('COM8',9600)
    while(True):
        ret, img = cap.read()
        img = cv2.resize(img, (200, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img_fixed = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        # 读取图像并进行处理
        img_fixed = np.array(img_fixed)
        img_fixed += 1
        im = Variable(torch.from_numpy(img_fixed).float())
        out = mlp(im).numpy()
        i = out.argmax()#通过标签预测小车正确的运行方向
        bt.write(i)#控制小车运行
        time.sleep(0.4)#以0.4秒作为一次控制周期

    cap.release()

if __name__ == '__main__':
    get_data()
    x = input('开始运行? y/n')
    if x == 'y':
        pre()