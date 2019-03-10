# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from ch04.gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())  # 纪录x的变化轨迹

        grad = numerical_gradient(f, x)  # 梯度就是返回一个向量，(对x1的在x点的导数，对x2的在x点导数)
        print("gradient_descent： grad_" + str(i) + str(grad))
        x -= lr * grad
        print("gradient_descent： new_x_" + str(i) + str(x))

    return x, np.array(x_history)


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


init_x = np.array([-3.0, 4.0])

lr = 0.1  # 学习率
step_num = 20  # 迭代次数
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
plt.plot(x_history[:, 0], x_history[:, 1], 'o')  # 轨迹图

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
