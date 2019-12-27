import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
def CrossEntropyLoss(y_predict, y_target):
    return torch.sum(-y_target * torch.log(y_predict) - (1 - y_target) * torch.log(1 - y_predict))

class logisticmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

model = logisticmodel()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.00001)

D = torch.tensor(pd.read_csv('label.csv', header = None).values, dtype = torch.float)
x_dataset = D[:, 0:4].view(-1,4)
#x_dataset[:, 0], x_dataset[:, 1] = x_dataset[:,0]+x_dataset[:,1], x_dataset[:,2]+x_dataset[:,3]
#x_dataset = x_dataset[:,0:2]
y_dataset = D[:, 4].view(-1,1)
duichen = True
if duichen:
    for i in range(len(x_dataset)):
        if x_dataset[i, 0] < 0:
            x_dataset[i] = -x_dataset[i]
last_loss = 0
for i in range(200000):
    optimizer.zero_grad()
    y_predict = model(x_dataset)
    loss = CrossEntropyLoss(y_predict, y_dataset)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print('------\n',loss, y_predict[y_dataset == 1], y_predict[y_dataset == 0])

    if abs(last_loss - loss) < 0.0001:
        print(last_loss, loss)
        oz = [item > 0.5 for item in y_predict]
        print(np.sum([oz[i] == y_dataset[i]  for i in range(len(y_dataset))]), '/', len(y_dataset))
        break
    last_loss = loss

