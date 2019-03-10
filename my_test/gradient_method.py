# conding:utf-8
import numpy as np
import matplotlib.pyplot as plt


# 原函数
def function_1(x):
    # 二元二次函数，这里最小值点为(0,0)
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient(f, x):
    """采用中心差分计算数值微分
    :param f: 原函数
    :param x: 当前位置(x1,x2)
    :return: 返回当前位置梯度
    """
    h = 1e-4  # 0.0001，定义间隔
    grad = np.zeros_like(x)  # 初始化梯度grad=(0,0)

    for idx in range(x.size):  # 分别计算每个元素位的微分
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # 如对以一个元素f([x1+h,x2])

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 如对第一个元素grad(new_x1,0)
        x[idx] = tmp_val  # 还原值

    return grad


def gradient_descent(f, init_x, l_rate=0.01, step_num=100):  # 默认学习率和迭代函数
    """迭代计算梯度
    :param f: 原函数
    :param init_x: 给定初始位置
    :param l_rate: 学习率
    :param step_num: 迭代次数
    :return: 返回函数值更新结果，以及函数值的变化用于画图
    """
    x = init_x  # 初始值点
    x_history = []  # 纪录函数值变化轨迹

    for i in range(step_num):  # 迭代次数
        x_history.append(x.copy())  # 添加纪录函数值变化轨迹
        # print("x_history: " + str(x_history))

        grad = numerical_gradient(f, x)  # 计算当前位置的梯度
        x -= l_rate * grad  # 沿着梯度方向更新函数值
        print("step_num: " + str(i+1) + " grad:" + str(grad) + " x:" + str(x))

    return x, np.array(x_history)


init_x = np.array([-3.0, 4.0])  # 初值点
l_rate = 0.1  # 学习率
step_num = 20  # 迭代次数

# 返回所计算的最小值，以及函数值变化轨迹
x, x_history = gradient_descent(function_1, init_x, l_rate=l_rate, step_num=step_num)

plt.plot([-10, 10], [0, 0], '--b')  # 坐标
plt.plot([0, 0], [-10, 10], '--b')

plt.plot(x_history[:, 0], x_history[:, 1], 'rx')  # x_history=(10, 2)

plt.xlim(-4, 4)
plt.ylim(-5, 5)
plt.xlabel('x0')
plt.ylabel('x1')
plt.show()
