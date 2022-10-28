# 一.对于全连接层的理解
全连接层的每一个结点都与上一层的所有结点相连，用来把前边提取到的特征综合起来。由于其全相连的特性，一般全连接层的参数也是最多的。
## 全连接层的前向计算
下图中连线最密集的2个地方就是全连接层，这很明显的可以看出全连接层的参数的确很多。在前向计算过程，也就是一个线性的加权求和的过程，全连接层的每一个输出都可以看成前一层的每一个结点乘以一个权重系数W，最后加上一个偏置值b得到，即 。如下图中第一个全连接层，输入有50*4*4个神经元结点，输出有500个结点，则一共需要50*4*4*500=400000个权值参数W和500个偏置参数b。  


![image](https://user-images.githubusercontent.com/114986300/196958095-ee9a13df-90d1-4c26-b4ee-1684a07c48a0.png)  

用一个简单的网络具体介绍一下推导过程  

![image](https://user-images.githubusercontent.com/114986300/196959064-c6d969ab-5dcc-4fee-b9cc-c763310e7ec8.png)  

可以写成如下矩阵形式：  

![image](https://user-images.githubusercontent.com/114986300/196959029-cb43d207-73d1-4ca7-a66a-1cc89ccd7a39.p

## 全连接层的反向传播
![image](https://user-images.githubusercontent.com/114986300/196961989-815cc341-a62d-4a02-a85b-902e7271625f.png)  

需要对W和b进行更新，还需要向前传递梯度，因此，需要计算三个偏导数：  
1）对上一层的输出求导   
2）对权重系数W求导    
3）对偏执系数b求导

全连接层的意义
连接层实际就是卷积核大小为上层特征大小的卷积运算，卷积后的结果为一个节点，就对应全连接层的一个点。
例如最后一个卷积层的输出为77512，连接此卷积层的全连接层为114096  
1）共有4096组滤波器  
2）每个滤波器有512个卷积核  
3）每个卷积核的大小为77  
4）则输出为11*4096

## 在pytorch中使用矩阵乘法实现全连接层
```
# 在pytorch中使用矩阵乘法实现全连接层
import torch.nn as nn
import torch
class FC(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(FC,self).__init__()
        self.w = torch.nn.Parameter(torch.randn(input_dim,output_dim))
        self.b = torch.nn.Parameter(torch.randn(output_dim))
        
    def forward(self,x):
        print("self.w  ",self.w)
        print("self.w.t()  ",self.w.t())
        y = torch.matmul(self.w.t(),x) + self.b  # torch.t()  求矩阵的转置的函数
        return y
linear = FC(3,2)
x = torch.ones(3)
y = linear(x)
print(y)
print(y.shape)

```

# 二.torch_test
## 2.1 TensorDataset
TensorDataset 可以用来对 tensor 进行打包。该类通过每一个 tensor 的第一个维度进行索引。因此，该类中的 tensor 第一维度必须相等。  
对于给定的Tensor数据（样本和标签），将它们包装成dataset。如果时numpy的array需要先转成Tensor。
```
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66, 44, 55, 66])
train_ids = TensorDataset(a, b)
```

## 2.2 常用随机数
### 1、torch.normal()  正态分布
返回一个张量，包含了从指定均值mean和标准差std的离散正态分布中抽取的一组随机数。
### 2、torch.rand()  均匀分布
### 3、torch.randint()  均匀分布
产生一个tensor，tensor中的每个元素都是从[low, high)(前闭后开)中获得的。使用均匀分布。
## 2.3 Pytorch中iter(dataloader)的使用
dataloader本质上是一个可迭代对象，可以使用iter()进行访问，采用iter(dataloader)返回的是一个迭代器，然后可以使用next()访问。
也可以使用enumerate(dataloader)的形式访问。

## 2.4 优化器
深度学习五个步骤：数据 ——> 模型 ——> 损失函数 ——> 优化器 ——> 迭代训练，通过前向传播，得到模型的输出和真实标签之间的差异，也就是损失函数，有了损失函数之后，模型反向传播得到参数的梯度，接下来就是优化器根据这个梯度去更新参数。
