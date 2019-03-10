# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    # relu函数
    return np.maximum(0, x)


def sigmoid(x):
    # sigmoid函数
    return 1 / (1 + np.exp(-x))


def step_function(x):
    # 阶跃函数
    return np.array(x > 0, dtype=np.int)  # 先计算bool值，再转成int


def show(x, y, ylim):
    # 画图
    plt.plot(x, y)
    plt.ylim(ylim)
    plt.show()  # plot在内存画，show一次性将内存的显示出来


def show_relu(x):
    # 展示relu函数图像
    y = relu(x)
    ylim = (-1.0, 5.5)  # y轴的范围，比输入的大d
    show(x, y, ylim)


def show_sigmoid(x):
    # 展示sigmoid函数图像
    y = sigmoid(x)
    ylim = (-0.1, 1.1)
    show(x, y, ylim)


def show_step(x):
    # 展示阶跃函数图像
    y = step_function(x)
    ylim = (-0.1, 1.1)
    show(x, y, ylim)


def show_sig_step_compare(x):
    # 对比阶跃函数和sigmoid函数图像
    y_sig = sigmoid(x)
    y_step = step_function(x)
    plt.plot(x, y_sig)
    plt.plot(x, y_step, 'k--')
    plt.ylim(-0.1, 1.1)
    plt.show()


x = np.arange(-5.0, 5.0, 0.1)  # x范围

show_relu(x)
show_sigmoid(x)
show_step(x)
show_sig_step_compare(x)
