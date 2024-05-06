---
authors:
    - mingkun
categories:
    - 深度学习
date: 2024-05-06
tags:
    - 深度学习
    - 工作流程
slug:  "MLworkflow"
---
# 深度学习工作流程（基于PyTorch）

一个完整的深度学习工作流程涉及处理数据、创建模型、优化参数和保存模型等步骤。

这篇blog提供了一个在FashionMNIST数据集上训练的一个图片分类网络实例，提供了一个较为规范的工作流程模板。

<!-- more -->

## 处理数据
PyTorch有两个处理数据的方法：Torch.utils.data.DataLoader和Torch.utils.data.Dataset。Dataset存储了样本及其相应的标签，而DataLoader则围绕Dataset包装了一个可迭代数据容器。
```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# 下载训练集
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)

# 下载测试集
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64  # 设置批量大小

# 创建data_loaders，可迭代数据集，迭代的每一个元素将返回一个批次，包括批量大小个元素的特征和标签
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
```
## 创建模型
```python
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):  # 初始化函数
        super(NeauralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):  # 前向传播函数
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NearalNetwork().to(device)
```

## 优化参数
```python
# 定义损失函数，评价模型预测结果与真实结果间的差距
loss_fn = nn.CrossEntropyLoss()

# 定义优化器，调整参数来缩小预测结果和真实结果的差距
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) 
    model.train() # 设置为训练模式（optimizing enable）
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 误差反传
        optimizer.step() # 一步更新

        # 固定批次打印训练信息
        if batch % 100 == 0
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

# 定义测试函数
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batchs = len(dataloader)
    model.eval() # 设置为测试模式（optimizing disable）
    test_loss, correct = 0, 0
    with torch.no_grad(): # 不计算梯度，节省空间
        for X, y in dataloader:
            X, y = X.to_device(), y.to_device()
            pred = model(X)
            test_loss += loss_fn(pred, y).item() # tensor.item()转换成标量
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # 计算正确预测个数        
    test_loss /= num_batches # 单批次平均训练损失
    correct /= size # 精确度
    print(f'Test Error: \n Accuracy: {*(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

# 训练过程通过几个epoch进行，模型在每一个epoch中进行参数更新
# 我们希望看到准确度在每个epoch中增加，损失在每个epoch中减少
epochs = 5
for t in range(epochs):
    print(f'Epoch {t+1}\n-----------------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

## 保存模型
```python
# 序列化内部状态字典（包含模型参数）
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model state to model.pth")
```

## 载入模型并预测
```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

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

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

## 总结
以上内容展示了完成一项深度学习基本任务所需要的全部工作流程, 在完成一项实际任务时，大致需要经过一下几个步骤：

1. 确定数据对象，选择合适的批量大小，创建Data Loader完成数据的批量预加载（一般分为训练集和测试集，实际问题中为了调参往往会增设k折交叉验证）。

2. 根据问题需要构造合适的网络模块，完成模型声明与实例化。

3. 选择合适的损失函数和优化器，在合适的迭代次数内对模型进行参数更新，并不断通过验证集准确率与损失大小确定模型是否处于学习状态（即是否未发生欠拟合/过拟合）。

4. 保存模型内部参数（保存序列化内部状态字典），需要使用时实例化模型并加载参数，即可进行预测任务。