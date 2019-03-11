# coding:utf-8
# 基于mnist数据集和随机梯度下降法实现简单神经网络

import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import datetime
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist


class TwoLayerNet:
    # 两层神经网络，初始化、预测、计算损失、计算准确率、定义向前传播梯度法、数字梯度计算
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 自动生成权重参数
        self.params = {}
        self.params['W1'] = np.random.rand(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)  # b初始化为0
        self.params['W2'] = np.random.rand(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def cross_entropy_error(self,y,t):
        """交叉熵损失
        :param y:
        :param t:
        :return:
        """


    def loss(self, x, t):
        # 定义损失函数
        y = self.predict(x)
        # 计算交叉熵损失函数
        return self.cross_entropy_error(y, t)

    def forward_gradient(self, x, t):
        """向前梯度
        :param x: 传入待训练图像
        :param t: 图像标签
        :return: 返回基于x，t训练得到的参数梯度
        """

        # 计算各个参数的梯度
        loss_W = lambda W: self.loss(x, t)

        # 计算梯度
        grads = {}
        grads['W1'] = self.numerical_gradient(loss_W, self.params['W1'])
        grads['W2'] = self.numerical_gradient(loss_W, self.params['W2'])
        grads['b1'] = self.numerical_gradient(loss_W, self.params['b1'])
        grads['b1'] = self.numerical_gradient(loss_W, self.params['b2'])

        return grads

    def numerical_gradient(f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        # readwrite：指定读写（ read-write）或只写（write-only）模式
        # multi_index：表示输出元素的索引
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])  # 迭代器，遍历W数组

        while not it.finished:
            idx = it.multi_index  # 迭代次序为(0,0)->(0,1)->(0,2)->(1,0)->(1,1)->(1,2)
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val  # 还原值
            it.iternext()  # 表示进入下一次迭代，如果不加这一句的话，输出的结果就一直都是(0, 0)。

        return grad


def train_neuralnet():
    print("training neuralnet...")
    # 数据读入
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    # 初始化模型
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    # 初始化超参数
    iters_num = 1000  # 迭代次数 10000
    learning_rate = 0.01  # 学习率

    # min_batch大小
    train_size = x_train.shape(0)
    batch_size = 100  # 每次训练100张

    # 纪录损失和准确率
    train_lost_list = []
    train_acc_list = []
    test_acc_list = []

    # epoch大小，大于一
    iter_per_epoch = max(train_size / batch_size, 1)

    # 模型迭代训练
    def run():  # 开始训练
        for i in range(iters_num):
            # 随机选取batch大小数据
            batch_mask = np.random.choice(train_size, batch_size)  # 在train_size中选batch_size个
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # 计算梯度
            grad = network.gradient()

    run()


if __name__ == '__main__':
    train_neuralnet()
