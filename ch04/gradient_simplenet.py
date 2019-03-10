# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 初始化参数

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


x = np.array([0.6, 0.9])  # 输入
t = np.array([0, 0, 1])  # 正确结果

net = simpleNet()

f = lambda w: net.loss(x, t)  # 在W参数空间内，求损失函数最小化
dW = numerical_gradient(f, net.W)  # 通过每次只求一个元素位的办法得到W每个位的损失的微分。由此得到梯度
# 梯度的维数和参数的维数是一样的，参数的维数就是参数的个数

print(net.W)
print(dW)  # 根据这个梯度来修改参数值。通常定义一个阈值，损失函数小于多少停止训练
