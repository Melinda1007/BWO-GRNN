# -*- coding: utf-8 -*-

import math

import numpy as np


def load_data(filename, num_out=1):
    """
    导入数据
    :param filename: 训练数据路径
    :param num_out: 输出值的个数，也就是倒数第几个开始为输出值
    此处输出为1，因为是一个输出
    :return:
    这部分的np.mat(label).T原先没有转置
    """
    f = open(filename)
    feature = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split(' ')
        for i in range(len(lines) - num_out):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
        label.append([float(x) for x in lines[-num_out:]])
    return np.mat(feature), np.mat(label)


def distance(X, Y):
    """
    计算两个样本之间的距离
    如果有多维数据，那就对应位置相减，再计算平方，然后压缩列求和
    :param X: 输入值
    :param Y: 输出值
    :return: 对应距离
    """
    return np.sqrt(np.sum(np.square(X - Y), axis=1))


def distance_mat(trainX, testX):
    """
    计算待测试样本与所有训练样本的欧式距离
    :param trainX: 训练样本
    :param testX: 测试样本
    :return: Euclidean_D(mat):测试样本与训练样本的距离矩阵
    """
    m, n = np.shape(trainX)
    p = np.shape(testX)[0]
    euclidean_d = np.mat(np.zeros((p, m)))
    for i in range(p):
        for j in range(m):
            euclidean_d[i, j] = distance(testX[i, :], trainX[j, :])[0, 0]
    return euclidean_d


def gauss(euclidean_d, sigma):
    """
    测试样本与训练样本的距离矩阵对应的Gauss矩阵
    :param euclidean_d: 测试样本与训练样本的距离矩阵
    :param sigma: Gauss函数的标准差
    :return: Gauss矩阵
    input:Euclidean_D(mat):测试样本与训练样本的距离矩阵
          sigma(float):Gauss函数的标准差
    output:Gauss(mat):Gauss矩阵
    """
    m, n = np.shape(euclidean_d)
    Gauss = np.mat(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            Gauss[i, j] = math.exp(- euclidean_d[i, j] / (2 * (sigma[j] ** 2)))
    return Gauss


def sum_layer(Gauss, trY):
    """求和层矩阵，列数等于输出向量维度+1,其中0列为每个测试样本Gauss数值之和"""
    m, l = np.shape(Gauss)  # X的维度和样本个数
    n = np.shape(trY)[1]  # 输出值的维度
    sum_mat = np.mat(np.zeros((m, n + 1)))
    # 对所有模式层神经元输出进行算术求和
    for i in range(m):
        sum_mat[i, 0] = np.sum(Gauss[i, :], axis=1)  # sum_mat的第0列为每个测试样本Gauss数值之和
    # 对所有模式层神经元进行加权求和，yij为模式层中第i个节点对应样本输出值的第j个输出值
    for i in range(m):
        for j in range(n):
            total = 0.0
            for s in range(l):
                total += Gauss[i, s] * trY[s, j]
            sum_mat[i, j + 1] = total  # sum_mat的后面的列为每个测试样本Gauss加权之和
    return sum_mat


def output_layer(sum_mat):
    """
    输出层输出
    :param sum_mat: 求和层输出矩阵
    :return: 输出层输出矩阵
    """
    m, n = np.shape(sum_mat)
    output_mat = np.mat(np.zeros((m, n - 1)))
    for i in range(n - 1):
        output_mat[:, i] = sum_mat[:, i + 1] / sum_mat[:, 0]
    return output_mat


def mean_square_error(y_output, y_test):
    """
        计算错误率的部分
        :param sum_mat: 求和层输出矩阵
        :return: 错误率
        """
    y1k, y1d = y_output.shape
    assert y_output.shape[0] == y_test.shape[0], 'Diff size'
    # 断言函数，判断真假，若等式不成立则输出错误信息
    diff = (y_output - y_test)  # 对应位置相减
    sum_diff = np.sum(diff, axis=1)  # 把所有输出值对应位置的误差相加
    square_diff = np.square(sum_diff)
    error = (np.sum(square_diff, axis=0) / y1d).tolist()
    return error[0][0]


class Grnn():
    def __init__(self, path, num_node, num_y):
        self.nodes = num_node
        self.path = path
        self.numy = num_y
        self.trX = None
        self.trY = None
        self.teX = None
        self.teY = None

    def prepare_data(self):
        # 1.导入数据
        feature, label = load_data(self.path, self.numy)
        # 2.数据集和测试集
        assert len(feature) > self.nodes, \
            "You only have {} samples, but you set {}".format(len(feature), self.nodes)

        self.trX = feature[0:self.nodes, :]
        self.trY = label[0:self.nodes, :]
        self.teX = feature[self.nodes:self.nodes + 10, :]
        self.teY = label[self.nodes:self.nodes + 10, :]

    def run(self, sigma=None, printmat=False):
        """
        计算新的光滑因子对应的误差值
        设计不同于GRNN网络的部分（蚁群算法通过这部分作为调节的适应度部分）
        :param sigma: 新的光滑因子
        :return: 误差值
        """
        # 3.模式层输出
        euclidean_d = distance_mat(self.trX, self.teX)
        # 创建sigma向量
        gauss_mat = gauss(euclidean_d, sigma)
        # 4.求和层输出
        sum_mat = sum_layer(gauss_mat, self.trY)
        # 5.输出层输出
        output_mat = output_layer(sum_mat)
        if printmat:
            print(output_mat)
        error = mean_square_error(output_mat, self.teY)
        if printmat:
            print("Error: {}\n".format(error))
        return error


if __name__ == '__main__':
    path = 'sine.txt'
    # 此处的路径是示例处理训练集的数据的位置
    samples = 40
    grnn = Grnn(path, samples, 3)
    grnn.prepare_data()  # 设置输出值Y的个数，这里默认为1，如果有两个设置2
    sigma = np.full((samples,), 0.35)
    # 设置处理测试集数据的位置
    # 此处的问题是：多维的输入测试集设计
    grnn.teX = np.mat([[146.30, -3.70, 152.92, 2.92, 290.60]])
    grnn.teY = np.mat([[-9.4, 230, 234.24]])
    error = grnn.run(sigma, True)
