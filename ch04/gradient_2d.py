# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):  # f为原函数，x为坐标值
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 初始化梯度，大小相同，元素都为0. x=(324,)

    for idx in range(x.size):  # 遍历x的每个元素，计算每个点的导数值
        # 先计算第一个元素位的导数，其中其他位为给定值
        tmp_val = x[idx]
        # 下面计算x左右的函数值
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)。计算x处的函数值,x是一个向量不是元素值

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 数值计算，中心差分求导数

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient(f, X):  # X=(2, 324)维度数组
    if X.ndim == 1:
        print("numerical_gradient: x.dim==1")  # 只有一行
        return _numerical_gradient_no_batch(f, X)
    else:
        print("numerical_gradient: x.dim==2")

        grad = np.zeros_like(X)  # 生成与X形状相同的数组

        for idx, x in enumerate(X):  # 返回两个序列，idx0=X,idx1=Y。idx指x第id个元素。这里遍历计算各个元素在各个坐标下的导数
            grad[idx] = _numerical_gradient_no_batch(f, x)  # 分别求两个方向的梯度,x=(324,)
            # print("numerical_gradient: idx,x= " + str(idx) + "," + str(x))
        return grad


def function_2(x):  # 二元二次方程y=x_0^2 + x_1^2
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)


# def tangent_line(f, x):
#     d = numerical_gradient(f, x)
#     y = f(x) - d * x
#     return lambda t: d * t + y


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)  # (18,)
    x1 = np.arange(-2, 2.5, 0.25)  # (18,)
    X, Y = np.meshgrid(x0, x1)  # 赋值语句，X=Y=np.meshgrid(x0, x1).X=Y=(18, 18)

    X = X.flatten()  # (324,),将18*18的二维矩阵压缩为324的一维矩阵
    Y = Y.flatten()  # (324,)

    grad = numerical_gradient(function_2, np.array([X, Y]))  # (2, 324)
    print(grad[0])
    plt.figure()
    # 画箭头，X, Y,是坐标.这里给出了反向梯度，就是下降最快方向
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
