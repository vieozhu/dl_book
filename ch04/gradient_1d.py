# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):  # 数值求导，中心差分
    # f是一个函数，指function_1，f(x)=function_1(x)
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):  # 待求导函数，原函数
    return 0.01 * x ** 2 + 0.1 * x


def tangent_line(f, x):
    d = numerical_diff(f, x)  # 在x点的函数值
    y = f(x) - d * x
    return lambda t: d * t + y  # lambda 参数:表达式。只有参数和表达式，因此函数匿名。表达式的值作为输出。
    # 这里返回一个函数，函数需要输入参数t


x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 10)  # tf返回值是一个函数，要求输入参数t。在x=5处相切
y2 = tf(x)

plt.plot(x, y, label='function_1')
plt.plot(x, y2, '--', label='tangent_line')
plt.legend()
plt.show()
