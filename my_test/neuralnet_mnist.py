# coding:utf-8
import sys, os

sys.path.append(os.pardir)  # 导入父目录，从而读取平行目录dataset
import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image


def img_show(x):
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    img = x_test[x]
    img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def sigmoid(x):
    # 对经过第一层的结果,使用激活函数sigmoid，定义一个阈值，切换输出
    return 1 / (1 + np.exp(-x))


def softmax(x):
    # 将结果转化为0-1的概率表示
    if x.ndim == 2:  # 维数
        x = x.T  # 转置
        x = x - np.max(x, axis=0)  # 取纵轴的最大值
        y = np.exp(x) / np.sum(np.exp(x), axis=0)  # axis=0沿着纵轴操作
        return y.T
    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def get_test_data():
    # 获取测试集图片和标签，用于检测参数效果
    # load_mnist函数使用了pickle功能，pickle可以将程序运行中的对象保存为文件，
    # 第二次读入可以立刻恢复之前运行的对象

    (data_train, label_train), (data_test, label_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False)
    return data_test, label_test


def init_network():
    # 调用训练好的参数w和b
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)  # 将pickle类型数据转换为python的数据结构。要求file参数为二进制只读文件rb
        # print(network)
        # 返回字典类型数据network{‘W1':([]),‘W2':([]),‘W3':([]),‘b1':([]),‘b1':([]),‘b1':([])}
    return network


def predict(network, data_test):  # 输入参数集，测试数据集
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 三层网络
    # print("data_test:" + str(np.shape(data_test)))
    # print("W1:" + str(np.shape(W1)))
    # print("b1:" + str(np.shape(b1)))
    # print("W2:" + str(np.shape(W2)))
    # print("b2:" + str(np.shape(b2)))
    # print("W3:" + str(np.shape(W3)))
    # print("b3:" + str(np.shape(b3)))

    a1 = np.dot(data_test, W1) + b1  # 第一层有50个神经元
    z1 = sigmoid(a1)  # 定义阈值，切换输出
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    # print("a1:" + str(a1))
    # print("z1:" + str(z1))
    # print("a2:" + str(a2))
    # print("z2:" + str(z2))
    # print("a3:" + str(a3))
    # print("y:" + str(y))

    # z1 = sigmoid(a1)
    return y  # 返回预测结果


data_test, label_test = get_test_data()
network = init_network()
accuracy_cnt = 0  # 预测正确的个数计数器

length = len(data_test)
for i in range(length):
    y = predict(network, data_test[i])
    p = np.argmax(y)  # 获取概率最高的元素的下标
    if p == label_test[i]:
        print(str(i) + " TRUE")
        accuracy_cnt += 1
    else:
        print("p=" + str(p) + " label=" + str(label_test[i]))
        print(str(i) + " FALSE")
        # img_show(i)

print("Accuracy: " + str(float(accuracy_cnt) / length))

# for i in range(len(data_test)):  # 遍历数据集的下标
