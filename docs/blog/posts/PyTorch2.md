---
authors:
    - mingkun
categories:
    - 深度学习
date: 2024-05-07
tags:
    - 张量
slug:  "tensor-overview"
---
# Tensor基础操作
PyTorch中，Tensor（张量）时操作数据的主要工具，它是一个包含单一类型的多维矩阵，与NumPy数组的形式类似，也共享底层内存位置。优点在于Tensor可以在GPU进行运算，亦可以实现自动梯度求解等操作。

这篇blog将介绍PyTorch框架下Tensor的一些基本操作。

<!-- more -->

## 头文件
```python
import torch
import numpy as np
```

## 张量初始化
可以通过直接初始化、从Numpy数组中初始化与从其它tensor中初始化来创建一个tensor
```python
# 直接初始化
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 从Numpy数组中初始化
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 从其它tensor中初始化
x_ones = torch.ones_like(x_data)  # 创建一个与x_data形状相同，且元素全部为1的向量(dtype=float32)
x_rand = torch.rand_like(x_data)  # 创建一个与x_data形状相同，且元素为(0, 1)区间内均匀分布抽样产生的随机数
x_randn = torch.randn_like(x_data) # 由标准正态分布抽样产生随机数

# 给定形状的初始化
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(x_data, x_np, x_ones, x_rand, x_randn, rand_tensor, ones_tensor, zeros_tensor)
```
效果
```
tensor([[1., 2.],
        [3., 4.]]) 
tensor([[1, 2],
        [3, 4]], dtype=torch.int32) 
tensor([[1., 1.],
        [1., 1.]]) 
tensor([[0.5376, 0.6459],
        [0.8904, 0.7159]]) 
tensor([[0.1786, 2.6272],
        [2.7045, 0.9933]]) 
tensor([[0.7170, 0.7762, 0.4273],
        [0.3025, 0.3687, 0.5089]]) 
tensor([[1., 1., 1.],
        [1., 1., 1.]]) 
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

## 查看张量属性
```python
tensor = torch.randn((3, 4))

tensor, tensor.shape, tensor.dtype, tensor.device
```
效果
```
tensor([[ 0.1958, -0.3580, -0.3055,  0.6545],
        [ 0.5811, -0.6416, -0.4026,  0.6289],
        [ 0.9658,  0.7655, -1.4409, -0.5016]])
(torch.Size([3, 4]), torch.float32, device(type='cpu'))
```

## 张量基本操作
将张量传递到GPU加速计算
```python
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print('yes')

tensor.device
```
效果
```
yes
device(type='cuda', index=0)
```
利用索引对张量进行剪裁、批量修改等操作
```python
print(tensor[0], tensor[:, 0], tensor[..., -1])
tensor[:, 1] = 0
print(tensor)
```
效果
```
tensor([ 0.1958, -0.3580, -0.3055,  0.6545], device='cuda:0'),
tensor([0.1958, 0.5811, 0.9658], device='cuda:0'),
tensor([ 0.6545,  0.6289, -0.5016], device='cuda:0')

tensor([[ 0.1958,  0.0000, -0.3055,  0.6545],
        [ 0.5811,  0.0000, -0.4026,  0.6289],
        [ 0.9658,  0.0000, -1.4409, -0.5016]], device='cuda:0')
```
在不同维度对张量进行拼接
```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
t2 = torch.cat([tensor, tensor, tensor], dim=0)
t1, t2
```
效果
```
(tensor([[ 0.1958,  0.0000, -0.3055,  0.6545,  0.1958,  0.0000, -0.3055,  0.6545,
           0.1958,  0.0000, -0.3055,  0.6545],
         [ 0.5811,  0.0000, -0.4026,  0.6289,  0.5811,  0.0000, -0.4026,  0.6289,
           0.5811,  0.0000, -0.4026,  0.6289],
         [ 0.9658,  0.0000, -1.4409, -0.5016,  0.9658,  0.0000, -1.4409, -0.5016,
           0.9658,  0.0000, -1.4409, -0.5016]], device='cuda:0'),
 tensor([[ 0.1958,  0.0000, -0.3055,  0.6545],
         [ 0.5811,  0.0000, -0.4026,  0.6289],
         [ 0.9658,  0.0000, -1.4409, -0.5016],
         [ 0.1958,  0.0000, -0.3055,  0.6545],
         [ 0.5811,  0.0000, -0.4026,  0.6289],
         [ 0.9658,  0.0000, -1.4409, -0.5016],
         [ 0.1958,  0.0000, -0.3055,  0.6545],
         [ 0.5811,  0.0000, -0.4026,  0.6289],
         [ 0.9658,  0.0000, -1.4409, -0.5016]], device='cuda:0'))
```
> tips: dim=0代表最高维度，dim增加，维数依次降低

线性运算
```python
x = torch.randn(4, 4)
# 矩阵乘法, y1, y2, y3结果相同，均完成了x与其转置矩阵相乘的操作
y1 = x @ x.T

y2 = x.matmul(x.T)

y3 = torch.rand_like(x)
torch.matmul(x, x.T, out=y3)
```
效果
```
tensor([[ 0.4628,  0.6459, -0.8538,  0.2622],
        [ 0.6459,  3.0141, -0.0512,  2.8879],
        [-0.8538, -0.0512, 11.1508,  1.5462],
        [ 0.2622,  2.8879,  1.5462,  4.0399]])
```
将位于不同设备的张量进行运算时，会发生错误
```python
y = tensor.matmul(x.T)
```
效果
```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[28], line 1
----> 1 y = tensor.matmul(x.T)

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, 
cuda:0 and cpu! (when checking argument for argument mat2 in method wrapper_CUDA_mm)
Selection deleted

```
对张量求和，会产生一个1x1张量，可以利用`tensor.item()`将1x1张量转化为标量
```python
agg = tensor.sum()
agg_item = agg.item()
agg_item, type(agg_item)
```
效果
```
(0.3755984306335449, float)
```
张量与np数组共享底层内存位置，改变一个将影响另一个
```python
t = torch.ones(5)
n = t.numpy()
print(t, n)

t.add_(1)
print(t, n)

np.add(n, 1, out=n)
print(t, n)
```
效果
```
tensor([1., 1., 1., 1., 1.]), array([1., 1., 1., 1., 1.], dtype=float32)
tensor([2., 2., 2., 2., 2.]), array([2., 2., 2., 2., 2.], dtype=float32)
tensor([3., 3., 3., 3., 3.]), array([3., 3., 3., 3., 3.], dtype=float32)
```

## 更多

从[PyTorch官方文档](https://pytorch.org/docs/stable/torch.html)了解更多tensor操作