# coding: utf-8
import sys, os

sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 1  # 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):  # 迭代更新参数值，得到最优参数。每次迭代对应一个准确率
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 梯度
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 每次计算损失，都要预测一遍，就要用参数predict一遍完整的网络，消耗计算资源，可以设定没10次或者n次纪录一次
    loss = network.loss(x_batch, t_batch)  # 用上一步更新了的参数，计算一次损失.结果为float(2.6)
    train_loss_list.append(loss)
    print("train_acc:", train_acc, "test_acc:", test_acc)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)  # 根据batch训练的参数，对完整数据测试一遍
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train_acc:", train_acc, "test_acc:", test_acc)
