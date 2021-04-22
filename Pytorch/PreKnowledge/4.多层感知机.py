#!/usr/bin/env python
# coding: utf-8

import torch
from torch.autograd import Variable
import torch.nn as nn

# 一定要继承 nn.Module
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
            我们在构建模型的时候，能够使用nn.Sequential的地方，尽量使用它，因为这样可以让结构更加清晰
        """
        super(TwoLayerNet, self).__init__()
        self.twolayernet = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        """
        在forward函数中，我们会接受一个Variable，然后我们也会返回一个Varible
        """
        y_pred = self.twolayernet(x)
        return y_pred


# M是样本数量，input_size是输入层大小
# hidden_size是隐含层大小，output_size是输出层大小
M, input_size, hidden_size, output_size = 64, 1000, 100, 10

# 生成随机数当作样本，同时用Variable 来包装这些数据，设置 requires_grad=False 表示在方向传播的时候，
# 我们不需要求这几个 Variable 的导数
x = Variable(torch.randn(M, input_size))
y = Variable(torch.randn(M, output_size))

model = TwoLayerNet(input_size, hidden_size, output_size)

# 定义损失函数
loss_fn = nn.MSELoss(reduction='sum')

## 设置超参数 ##
learning_rate = 1e-4
EPOCH = 300

# 使用optim包来定义优化算法，可以自动的帮我们对模型的参数进行梯度更新。这里我们使用的是随机梯度下降法。
# 第一个传入的参数是告诉优化器，我们需要进行梯度更新的Variable 是哪些，
# 第二个参数就是学习速率了。
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

## 开始训练 ##
for t in range(EPOCH):

    # 向前传播
    y_pred = model(x)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 显示损失
    if (t + 1) % 50 == 0:
        print(loss.data.item())

    # 在我们进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度。
    optimizer.zero_grad()

    # 计算梯度
    loss.backward()

    # 更新梯度
    optimizer.step()