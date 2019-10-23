---
layout: post
title:  "Deep Learning with PyTorch:A 60 Minute Blitz"
date:   2019-10-22 00:09:00 +0800
tags: deeplearning
color: rgb(123,45,45)
cover: '../assets/pytorch.jpg'
subtitle: 'pytorch官方教程'
---

# Deep Learning with PyTorch:A 60 Minute Blitz

## 1. What is pytorch

它是一个基于python的科学计算包，有两个目的：

+ 替代NumPy在GPU上的功能
+ 提供最大复杂度和速度的深度学习研究平台

### Tensors(张量)

Tensors 类似于NumPy的ndarrays，还可以支持在GPU上被用于加速运算

~~~python
from __future__ import print_function
import torch
~~~

> 注：上一句话指为了避免某特性和当前版本不兼容，就从future模块中导入，一般指将py2中的某功能替换成py3中的函数，例如本句指 print_function：
> ~~~python
> #python2.7
> print "hello"
> #python3.6
> print("hello")
> ~~~

建立5X3的矩阵：

~~~python
x = torch.empty(5, 3)#注：当一个未被初始化的矩阵被声明时，矩阵元素可能是任何一个浮点数
print(x)
~~~
输出：
~~~python
tensor([[1.0653e-38, 1.0469e-38, 9.5510e-39],
        [1.0745e-38, 9.6429e-39, 1.0561e-38],
        [9.1837e-39, 1.0653e-38, 8.4490e-39],
        [8.9082e-39, 8.9082e-39, 1.0194e-38],
        [9.1837e-39, 1.0469e-38, 1.0286e-38]])
~~~
~~~python
x = torch.rand(5, 3)#0到1之间的随机数
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
~~~
输出：
~~~python
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
~~~
~~~python
x = torch.tensor([5.5, 3])#直接赋值
print(x)
~~~
输出：
~~~python
tensor([5.5000, 3.0000])
~~~

~~~
x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.double)#new方法占用空间
print(x)

x = torch.randn_like(x, dtype=torch.float)#重写dtype,size保持不变
print(x)
~~~

输出：

~~~python
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[ 0.4085,  1.7913,  1.2974],
        [-1.2097, -0.2682, -0.1624],
        [ 0.2719, -0.1450, -1.2968],
        [ 0.2282,  0.8241,  0.0784],
        [ 1.0826,  0.2281,  1.6897]])
~~~

~~~python
print(x.size())#输出维度，输出的维度的类型为tuple(元组）
~~~

输出：

~~~python
torch.Size([5, 3])
~~~

### 计算操作

计算操作的语法有很多种

#### 加法：

~~~python
x = torch.rand(5, 3)
y = torch.rand(5, 3)

print(x+y)

print(torch.add(x,y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)# 加法:输出一个新的张量变量
print(result)

y.add_(x)# 加法：加至y上
print(y)
~~~

> 注：任何一个会改变当前张量的函数都会有一个`_`后缀 例如：`x.copy_(y)` 和`x.t_()`都将使`x`改变

#### 索引

可以使用NumPy中的索引操作

~~~python
print(x[:, 1])
~~~

输出：

```python
tensor([0.8807, 0.3227, 0.2010, 0.3679, 0.6297])
```

#### 改变张量的大小

可以使用`torch.view`来改变一个张量的大小和形状

~~~python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # -1指这个维度是由另一个维度推断出的 即2=16/8
print(x.size(), y.size(), z.size())
~~~

输出：

~~~python
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
~~~

#### item
如果有一个仅具有一个元素的张量，用`.item()`来得到一个python数字类型的值

~~~python
x = torch.randn(1)
print(x)
print(x.item())
~~~

输出：

~~~python
tensor([0.6313])
0.6313426494598389
~~~

更多的计算操作[operation](https://pytorch.org/docs/stable/torch.html)

### NumPy 关联

numpy中的array数组可以和torch中的tensor张量互相转换。

如果torch中的tensor在cpu上时，两者甚至会共享存储空间，改变其中一个会导致另一个的改变。

+ tensor->array

~~~
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)
~~~

输出：

~~~python
tensor([1., 1., 1., 1., 1.])
[1. 1. 1. 1. 1.]
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
~~~
+ array->tensor

~~~python
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
~~~

输出：

~~~python
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
~~~

cpu上的所有张量（除了CharTensor）都可以和numpy互相转换

### CUDA Tensors

Tensors 可以用`.to`方法移动到任何设备上

~~~
# 仅当cuda设备存在时候才可以运行
# 用 ``torch.device`` 对象来移动tensors出入gpu
if torch.cuda.is_available():
    device = torch.device("cuda")          # 一个cuda设备对象
    y = torch.ones_like(x, device=device)  # 直接在cpu上创建tensor
    x = x.to(device)                       # 也可以使用 x = x.to(cuda)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # 同时可被用作转换dtype
~~~

输出：

~~~python
tensor([[1.4424, 1.9343, 1.4623],
        [1.1653, 1.5065, 1.2500],
        [1.7856, 1.5435, 1.6151],
        [1.2938, 1.5354, 1.5707],
        [1.7895, 1.5344, 1.0466]], device='cuda:0')
tensor([[1.4424, 1.9343, 1.4623],
        [1.1653, 1.5065, 1.2500],
        [1.7856, 1.5435, 1.6151],
        [1.2938, 1.5354, 1.5707],
        [1.7895, 1.5344, 1.0466]], dtype=torch.float64)
~~~

注意要下载对应版本且可在gpu上使用的cuda，这里就不再赘述了。

## 2. AUTOGRAD: AUTOMATIC DIFFERENTIATION

