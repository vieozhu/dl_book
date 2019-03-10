# coding: utf-8
import numpy as np


def _numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad


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
