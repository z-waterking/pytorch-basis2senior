# -*- coding: utf-8 -*-#
'''
@Project    :   DeepLearning
@File       :   多层感知机.py
@USER       :   ZZZZZ
@TIME       :   2021/4/21 11:29
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# ------------------------------------- 加载数据 -------------------------------------
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="../Data",
    train=True,
    download=True,
    transform=ToTensor(), # 图像转为tensor
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="../Data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 检验数据集
for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# ------------------------------------- 创建模型 -------------------------------------
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        # 必须要调用父类的方法
        super(NeuralNetwork, self).__init__()
        # 将tensor打平操作
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print("model:", model)

# ------------------------------------- 设置loss -------------------------------------
loss_fn = nn.CrossEntropyLoss()

# ------------------------------------- 反向传播 -------------------------------------
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# ------------------------------------- 训练模型 -------------------------------------
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        # 前向传播
        pred = model(X)
        # 计算loss
        loss = loss_fn(pred, y)

        # Backpropagation
        # 把梯度置为0
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()

        # 输出log
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# ------------------------------------- 测试模型 -------------------------------------
def test(dataloader, model):
    size = len(dataloader.dataset)
    # 不启用 BatchNormalization 和 Dropout
    model.eval()

    test_loss, correct = 0, 0
    # 不要track梯度
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)
    print("Done!")

    # ------------------------------------- 保存模型 -------------------------------------
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # ------------------------------------- 加载模型 -------------------------------------
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))

    # ------------------------------------- 加载模型测试 -------------------------------------
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # 不启用 BatchNormalization 和 Dropout
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')