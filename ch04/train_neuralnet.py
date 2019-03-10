# coding: utf-8
# mini-batch随机梯度下降法
import sys, os
import datetime

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from ch04.two_layer_net import TwoLayerNet

start_time = datetime.datetime.now()
print("start_time: " + str(start_time.strftime('%H:%M:%S.%f')))
print()

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)  # 图像大小784，隐藏层神经元个数50，输出层10个数字

iters_num = 10000  # 适当设定迭代的次数10000
train_size = x_train.shape[0]
batch_size = 100  # min_batch 100
learning_rate = 0.1  # 0.1

# 纪录损失函数、训练准确率、测试准确率变化情况
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)  # 每个epoch，遍历一遍所有数据（随机batch不一定遍历所有图片）

for i in range(iters_num):  # n次迭代开始
    # min_batch随机选取batch张图片，所以这些图片都被重复学习过(train_size / batch_size)次
    # np.random.choice从train_size中随机选取batch_size个，未指定概率则采用一致分布
    batch_mask = np.random.choice(train_size, batch_size)  # (100,)。得到的是下标

    x_batch = x_train[batch_mask]  # (100, 784)
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 纪录损失函数
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)  # 每次迭代都纪录损失函数

    if i % iter_per_epoch == 0:  # 每个epoch计算一次准确率
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        now_time = datetime.datetime.now().strftime('%H:%M:%S.%f')
        print(str(now_time) + " | train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制图形(左边损失图像右边准确率图像)
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
# y = np.arange(len(train_loss_list))
# plt.plot(y, train_loss_list, label='train_loss')

plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 结束时间
end_time = datetime.datetime.now()
total_time = (end_time - start_time).seconds
print()
print("end_time: " + str(end_time.strftime('%H:%M:%S.%f')))
print("total_time: " + str(total_time) + "(s)")
