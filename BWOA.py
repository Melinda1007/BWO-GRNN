import math
from itertools import chain
from random import uniform
from random import choice
from random import random
from random import randint
from copy import deepcopy
import sys
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error,r2_score

import unicodedata

import gol

# matpoltlib库相关
# 用于绘制图表
# from pylab import *
# import matplotlib.pyplot as plt
#
# gol._init()

import matplotlib.pyplot as plt

import numpy as np
import math

# 程序退出有Error但是没有回溯
from bwo.main import _generate_new_position

sys.tracebacklimit = 0

# 定义grnn的函数部分

def load_data(filename, num_out=1):
    """
    导入数据
    :param filename: 训练数据路径
    :param num_out: 输出值的个数，也就是倒数第几个开始为输出值
    此处输出为1，因为是一个输出
    这里设置是为了方便改变数值
    :return:
    这部分的np.mat(label).T原先没有转置
    """
    # 此处是自建数据集（3列）部分用的导入程序
    # f = open(filename)
    # feature = []
    # label = []
    # for line in f.readlines():
    #     feature_tmp = []
    #     lines = line.strip().split('\t')
    #     # lines = str.strip("\t")
    #     for i in range(len(lines) - num_out):
    #         feature_tmp.append(float(lines[i]))
    #     feature.append(feature_tmp)
    #     # label.append([float(x) for x in lines[-num_out:]])
    #     # linex = [' '.join([i.strip() for i in x.strip().split('\t')]) for x in lines]
    #     label.append(float(lines[-1]))
    #     # label.append([float(x) for x in lines[-num_out:]])
    # return np.mat(feature), np.mat(label).T

    # # sine.txt
    # f = open(filename)
    # feature = []
    # label = []
    # for line in f.readlines():
    #     feature_tmp = []
    #     lines = line.strip().split(' ')
    #     for i in range(len(lines)-1):
    #         feature_tmp.append(float(lines[i]))
    #     feature.append(feature_tmp)
    #     label.append(float(lines[-1]))
    # return np.mat(feature), np.mat(label).T

    # f = open(filename)
    # feature = []
    # label = []
    # for line in f.readlines():
    #     feature_tmp = []
    #     lines = line.strip().split(' ') # 去掉换行符号，遇到空格则隔开
    #     for i in range(len(lines) - num_out):
    #         feature_tmp.append(float(lines[i]))
    #     feature.append(feature_tmp)
    #     label.append([float(x) for x in lines[-num_out:]])
    #     # label.append(float(lines[-1]))
    # return np.mat(feature), np.mat(label)

    f = open(filename)
    feature = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split('\t')
        for i in range(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
        label.append(float(lines[-1]))
    return np.mat(feature),np.mat(label).T

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
          此处sigma是矩阵
    output:Gauss(mat):Gauss矩阵
    """
    m, n = np.shape(euclidean_d)
    Gauss = np.mat(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            sigma1 = np.array(sigma)
            Gauss[i, j] = math.exp(- euclidean_d[i, j] / (12 * (sigma1 ** 2.5)))
    return Gauss

def sum_layer(Gauss,trY):
    """求和层矩阵，列数等于输出向量维度+1,其中0列为每个测试样本Gauss数值之和"""
    m, l = np.shape(Gauss)  # X的维度和样本个数
    n = np.shape(trY)[1]  # 输出值的维度
    sum_mat = np.mat(np.zeros((m, n+1)))  # 求和层要比这个多一个，则为n+1
    # 对所有模式层神经元输出进行算术求和
    for i in range(m):
        sum_mat[i,0] = np.sum(Gauss[i, :], axis = 1)  # sum_mat的第0列为每个测试样本Gauss数值之和
    # 对所有模式层神经元进行加权求和，yij为模式层中第i个节点对应样本输出值的第j个输出值
    for i in range(m):
        for j in range(n):
            total = 0.0
            for s in range(l):
                total += Gauss[i, s] * trY[s, j]
            sum_mat[i, j+1] = total           # sum_mat的后面的列为每个测试样本Gauss加权之和
    return sum_mat

def output_layer(sum_mat):
    """
    输出层输出
    input:sum_mat(mat):求和层输出矩阵
    output:output_mat(mat):输出层输出矩阵
    """
    m, n = np.shape(sum_mat)
    output_mat = np.mat(np.zeros((m, n - 1)))
    for i in range(n - 1):
        output_mat[:,i] = sum_mat[:, i + 1] / sum_mat[:, 0]
    return output_mat

# 增加求错误率的定义，目的是为了显示优化程度，但是不用于优化算法的调节
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
    # 此处error[0][0]代表的是第一个数字
    return error[0][0]

# records_real = []
# records_predict = []
#
#
# def fitness_func(records_real, records_predict):
#     """
#     均方根误差 估计值与真值 偏差
#     """
#     if len(records_real) == len(records_predict):
#         mse = sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
#         return math.sqrt(mse)
#     else:
#         return None

# 初次使用grnn，为了在数据集中找出最优sigma而设计的
class Grnn_gai():
    def __init__(self, path,  num_y):
        # self.nodes = num_node
        self.path = path
        self.numy = num_y
        self.trX = None
        self.trY = None
        self.teX = None
        self.teY = None
        self.output_mat = None

    def prepare_data(self):
        # 1.导入数据
        feature, label = load_data(self.path, self.numy)
        #print('------------------------1. Load Data----------------------------')
        # 2.数据集和测试集
        # assert len(feature) > self.nodes, \
        #     "You only have {} samples, but you set {}".format(len(feature), self.nodes)
        #print('--------------------2.Train Set and Test Set--------------------')
        self.trX = feature[0:250, :]
        self.trY = label[0:250, :]
        self.teX = feature[250:540, :]
        self.teY = label[250:540, :]

    # 这个版本是不包括错误率，仅返回输出矩阵
    def run(self, sigma=None, printmat=False):
        """
        计算新的光滑因子对应的误差值
        :param sigma: 新的光滑因子
        :return: 误差值
        """
        # 3.模式层输出
        #print('---------------------3. Output of Hidden Layer------------------')
        euclidean_d = distance_mat(self.trX, self.teX)
        # 创建sigma向量
        gauss_mat = gauss(euclidean_d, sigma)
        # 4.求和层输出
        #print('---------------------4. Output of Sum Layer---------------------')
        sum_mat = sum_layer(gauss_mat, self.trY)
        # 5.输出层输出
        output_mat = output_layer(sum_mat)
        #print("GRNN输出: {}\n".format(output_mat))
        return output_mat

# BWO算法部分，用于寻找最优sigma
class BWO:
    def __init__(self, argrs, root, bwo_grnn):
        """
        初始化算法的参数
        :param grnn: GRNN网络
        :param sigma_mat: 可行解矩阵，[1 x citys]
        :param maxiter 最大迭代次数
        """
        self.root = root
        #self.npop = bwo_grnn.samples
        self.grnn = bwo_grnn.grnn
        self.aval_sigma = bwo_grnn.aval_sigma_mat
        self.maxiter = bwo_grnn.max_iter
        self.PP = argrs.pp
        self.PM = argrs.pm
        self.CR = argrs.cr
        self.save = argrs.save
        # self.distance_graph = [[1.0 for col in range(self.city_num)] for raw in range(self.city_num)]
        # self.pheromone_graph = [[1.0 for col in range(self.ant_num)] for raw in range(self.city_num)]  # 50 x 20


        #self.best_sigma = []
        #self.min_error = []
        #self.final_sigma = []

        # self.__bindEvents()
        # self.new()
        # self.search_path()

    def _generate_new_position(x0: list = None, dof: int = None, bounds: list = None) -> list:  # 定义新的位置
        '''GENERATE NEW POSITION

        Parameters
        ----------
        dof : int
        x0 : list
        bounds : list of tuples [(x1_min, x1_max),...,(xn_min, xn_max)]

        Returns
        -------
        list

        Notes
        -----
        There are several ways in which an initial position can be generated.
        Outlined below are all possible scenarios and outputs.

        nomenclature:
            "dof" = "degrees of freedom" = "dimensions" = "d"
            p = new initial position vector of length d

        just bounds:
            for each position i in bounds,  p[i] = random value in [i_min, i_max]]

        just x0:
            for each position i in x0: p[i] = x0[i] + random value in [-1, 1]

        just dof:
            for each position i from 0 to d,  p[i] = random value in [-1, 1]

        dof + x0:
            since dof and x0 are redundent from a dimensionality perspective,
            this situation will defer to the case above "just x0".

        dof + bounds:
            since dof and bounds are redundent from a dimensionality perspective,
            this situation wll defer to the case above "just bounds"

        x0 + bounds:
            for each position i in x0:
                p[i] = x0[i] + random value in [-1, 1] constrained by bounds[i].min
                and bounds[i].max

        dof + x0 + bounds:
            see case: "x0 + bounds" above

        All this boils down to four cases (ordered by information gain from user):
        1) x0 and bounds
        2) bounds
        3) x0
        4) dof
        '''

        if x0 and bounds:
            return [min(max(uniform(-1, 1) + x0[i], bounds[i][0]), bounds[i][1]) for i in range(len(x0))]

        if bounds:
            return [uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]

        if x0:
            return [x_i + uniform(-1, 1) for x_i in x0]

        if dof:
            return [uniform(-1, 1) for _ in range(0, dof)]


    def functhing(sigma):
        path = 'Fin_nor.txt'
        grnn =Grnn_gai(path, 1)
        grnn.prepare_data()
        records_predict = grnn.run(sigma, True)
        records_real = grnn.teY
        if len(records_real) == len(records_predict):
            mse = sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
            #print(len(records_real), len(records_predict))
            #print("此时的sigma: {}\n".format(sigma))
            #print("func输出: {}\n".format(math.sqrt(mse)))
            return math.sqrt(mse)
        else:
            return None



    def minimize(func, x0=None, dof=None, bounds=None, pp=0.6, cr=0.44, pm=0.4,  # pp是繁殖率；CR是同类相食率；PM是突变率
                 npop=100, disp=False, maxiter=50):  # 定义最小化，最优化； maxiter：最大循环次数
        '''
        Parameters
        ----------
        x0 : list
            initial guess 初始猜测
        pp : float
            procreating percentage
        cr : float
            cannibalism rate. A cr of 1 results in all children surviving. A cr of 0
            results in no children surviving
        pm : float
            mutation rate

        Returns
        -------
        float : solution at global best
        list : position at global best

        References
        ----------
        '''

        # do some basic checks before going any further # 基本检查和设定
        assert type(disp) == bool, 'parameter: disp -> must be of type: bool'
        assert type(npop) == int, 'parameter: npop -> must be of type: int'
        assert type(maxiter) == int, 'parameter: maxiter -> must be of type int'
        if x0 is not None: assert type(x0) == list, 'x0 must be of type: list'
        if dof is not None: assert type(dof) == int, 'parameter: dof -> must be of type: int'
        if bounds is not None: assert type(bounds) == list, 'parameter: bounds -> must be of type list'
        assert x0 is not None or dof is not None or bounds is not None, 'must specify at least one of the following: x0, dof, or bounds'
        if x0 and bounds: assert len(bounds) == len(x0), 'x0 and bounds must have same number of elements'
        assert pp > 0 and pp <= 1, 'procreating percentage "pp" must be: 0 < pp <= 1'
        assert cr >= 0 and cr <= 1, 'cannibalism rate "cr" must be: 0 < cr <= 1'
        assert pm >= 0 and pm <= 1, 'mutation rate "pm" must be: 0 < pm <= 1'
        assert maxiter > 0, 'maxiter must be greater than zero.'

        # check bounds specification if necessary
        if bounds:
            assert type(bounds) == list, 'bounds must be of type: list'
            for b in bounds:
                assert type(
                    b) == tuple, 'element in bounds is not of type: tuple. ever every element must be a tuple as specified (v_min, v_max)'
                assert b[0] < b[1], 'element in bounds specified incorrectly. must be (xi_min, xi_max)'

        # constants
        if x0 is not None:
            dof = len(x0)
        elif bounds is not None:
            dof = len(bounds)

        nr = int(npop * pp)  # number of reproduction
        nm = int(npop * pm)  # number of mutation children
        spacer = len(str(npop))  # for logging only

        # initialize population 初始化种群
        #popyuan = [_generate_new_position(x0, dof, bounds) for _ in range(0, npop)]
        #print("此时的sigma合集: {}\n".format(popyuan))

        # main loop 主循环
        hist = []
        list1 = []
        global list2
        # 设置全局变量list2
        for epoch in range(0, maxiter):


            popyuan = [_generate_new_position(x0, dof, bounds) for _ in range(0, npop)]
            #print("此时的sigma合集: {}\n".format(popyuan))
            # initialize epoch 初始化过程
            pop = sorted(popyuan, key=lambda x: func(x), reverse=False)  # 排序（以func的大小排序，即使用适应度函数大小进行排序）
            # 此处pop变成适应度函数计算后排序的结果，生成新的列表
            pop1 = deepcopy(pop[:nr])
            pop2 = []
            pop3 = []
            gbest = pop[0]

            # print something useful 输出的是每一次的循环结果
            if disp: print(f'> ITER: {epoch + 1:>{spacer}} | GBEST: {func(gbest):0.6f}')

            # 此处的代码实现对每次寻找最优的gbest存储
            list1.append(gbest)
            # print(list1)
            list2 = list(chain.from_iterable(list1))

            # 计算相邻最优的差值的绝对值
            list_with_diff_fabs = []
            for n in range(1, len(list2)):
                difference = math.fabs(list2[n] - list2[n - 1])
                list_with_diff_fabs.append(difference)
            # print("Difference between adjacent elements in the list(||): \n", list_with_diff_fabs)
            if not list_with_diff_fabs == []:
                print(list_with_diff_fabs[-1])

            # procreation and cannibalism 繁殖和同类相食
            for i in range(0, nr):

                # randomly pick two parents 随机选择两个父母
                i1, i2 = randint(0, len(pop1) - 1), randint(0, len(pop1) - 1)
                p1, p2 = pop1[i1], pop1[i2]

                # crossover
                children = []
                for j in range(0, 1):  # 子代的产生过程发生Nvar/2次

                    # generate two new children using equation (1) 产生子代的公式
                    alpha = random()  # 生成一个（0，1）间的随机数
                    c1 = [(alpha * v1) + ((1 - alpha) * v2) for v1, v2 in zip(p1, p2)]
                    c2 = [(alpha * v2) + ((1 - alpha) * v1) for v1, v2 in zip(p1, p2)]

                    # persist new children to temp population
                    children.append(c1)  # 将产生的子代放入堆栈
                    children.append(c2)

                # cannibalism - destroy male; since female black widow spiders are
                # larger and often end up killing the male during mating, we'll
                # assume that the fitter partent is the female. thus, we'll delete
                # the weaker parent.
                # 由于交配后雄性蜘蛛会被吃掉，判断雌性蜘蛛的方式是强壮的一方是雌性蜘蛛，所以循环中消灭父辈中弱的一方
                if func(p1) > func(p2):
                    pop1.pop(i1)  # 根据适应度大小消灭雄性蜘蛛
                else:
                    pop1.pop(i2)

                # cannibalism - destroy some children
                # 根据同类相食原则杀掉一些子代（数量是同类相食率，标准是适应度）
                children = sorted(children, key=lambda x: func(x), reverse=False)
                children = children[:max(int(len(children) * cr), 1)]

                # add surviving children to pop2
                pop2.extend(children)

            # mutation 突变过程
            for i in range(0, nm):
                # pick a random child
                m = choice(pop2)

                # pick two random chromosome positions 选择两个染色体位置
                cp1, cp2 = randint(0, dof - 1), randint(0, dof - 1)

                # swap chromosomes 交换两个染色体
                m[cp1], m[cp2] = m[cp2], m[cp1]

                # persist 存储
                pop3.append(m)

            # assemble final population 更新种群（pop=pop2+pop3）
            pop2.extend(pop3)
            pop = deepcopy(pop2)

        # popx = list1
        # # 此处pop变成适应度函数计算后排序的结果，生成新的列表
        # popx = sorted(popx, key=lambda x: func(x), reverse=False)
        # pop1x = deepcopy(popx[:nr])
        # pop2x = []
        #
        # # procreation and cannibalism 繁殖和同类相食
        # for i in range(0, nr):
        #
        #     # randomly pick two parents 随机选择两个父母
        #     i1, i2 = randint(0, len(pop1x) - 1), randint(0, len(pop1x) - 1)
        #     p1, p2 = pop1x[i1], pop1x[i2]
        #
        #     # crossover
        #     children = []
        #     for j in range(0, 1):  # 子代的产生过程发生Nvar/2次
        #
        #         # generate two new children using equation (1) 产生子代的公式
        #         alpha = random()  # 生成一个（0，1）间的随机数
        #         c1 = [(alpha * v1) + ((1 - alpha) * v2) for v1, v2 in zip(p1, p2)]
        #         c2 = [(alpha * v2) + ((1 - alpha) * v1) for v1, v2 in zip(p1, p2)]
        #
        #         # persist new children to temp population
        #         children.append(c1)  # 将产生的子代放入堆栈
        #         children.append(c2)
        #
        #     # cannibalism - destroy male; since female black widow spiders are
        #     # larger and often end up killing the male during mating, we'll
        #     # assume that the fitter partent is the female. thus, we'll delete
        #     # the weaker parent.
        #     # 由于交配后雄性蜘蛛会被吃掉，判断雌性蜘蛛的方式是强壮的一方是雌性蜘蛛，所以循环中消灭父辈中弱的一方
        #     if func(p1) > func(p2):
        #         pop1x.popx(i1)  # 根据适应度大小消灭雄性蜘蛛
        #     else:
        #         pop1x.popx(i2)
        #
        #     # cannibalism - destroy some children
        #     # 根据同类相食原则杀掉一些子代（数量是同类相食率，标准是适应度）
        #     children = sorted(children, key=lambda x: func(x), reverse=False)
        #     children = children[:max(int(len(children) * cr), 1)]
        #
        #     # add surviving children to pop2
        #     pop2x.extend(children)
        # popx = deepcopy(pop2x)

        popx = sorted(list2, key=lambda x: func(x), reverse=False)
        sigmabest = popx[0]

        # return global best position and func value at global best position 输出返回全局最佳的位置和最佳的func值
        return func(gbest), gbest, sigmabest

# 第二次调用grnn，用于求出sigma之后正式计算过程
class Grnnyuc():
    def __init__(self, path,  num_y):
        # self.nodes = num_node
        self.path = path
        self.numy = num_y
        self.trX = None
        self.trY = None
        self.teX = None
        self.teY = None
        self.output_mat = None

    def prepare_data(self):
        # 1.导入数据
        feature, label = load_data(self.path, self.numy)
        #print('------------------------1. Load Data----------------------------')
        # 2.数据集和测试集
        # assert len(feature) > self.nodes, \
        #     "You only have {} samples, but you set {}".format(len(feature), self.nodes)
        #print('--------------------2.Train Set and Test Set--------------------')
        self.trX = feature[0:250, :]
        self.trY = label[0:250, :]
        self.teX = feature[250:540, :]
        self.teY = label[250:540, :]

    # 这个版本是不包括错误率，仅返回输出矩阵
    def run(self, sigma=None, printmat=False):
        """
        计算新的光滑因子对应的误差值
        :param sigma: 新的光滑因子
        :return: 误差值
        """
        # 3.模式层输出
        #print('---------------------3. Output of Hidden Layer------------------')
        euclidean_d = distance_mat(self.trX, self.teX)
        # 创建sigma向量
        gauss_mat = gauss(euclidean_d, sigma)
        # 4.求和层输出
        #print('---------------------4. Output of Sum Layer---------------------')
        sum_mat = sum_layer(gauss_mat, self.trY)
        # 5.输出层输出
        output_mat = output_layer(sum_mat)
        #print("GRNN输出: {}\n".format(output_mat))
        return output_mat

if __name__ == "__main__":
    fbest, xbest, sigmabest = BWO.minimize(BWO.functhing, bounds=[(0, 1)], maxiter=1, disp=True)
    # ll = list2
    print(sigmabest)

    # 计算出的最优sigma值放入grnn网络进行正式预测
    path = 'Fin_nor_3.txt'
    grnn = Grnnyuc(path, 1)
    grnn.prepare_data()
    records_predict = grnn.run(sigmabest, True)
    records_real = grnn.teY
    # plt.scatter(records_predict,records_real)

    if len(records_real) == len(records_predict):
        # mse = sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
        # # print(len(records_real), len(records_predict))
        # # print("此时的sigma: {}\n".format(sigma))
        # # print("func输出: {}\n".format(math.sqrt(mse)))
        # rmse = math.sqrt(mse)
        # mae = sum([abs(x - y) for x, y in zip(records_real, records_predict)]) / len(records_real)
        # r2 = abs(1 - mse / np.var(records_real))
        # r_2 = r2_score(records_real, records_predict)
        # plt.scatter(records_predict.tolist(), records_real.tolist())
        print("mean_absolute_error:", mean_absolute_error(records_real, records_predict))
        print("mean_squared_error:", mean_squared_error(records_real, records_predict))
        print("rmse:", math.sqrt(mean_squared_error(records_real, records_predict)))
        print("r2 score:", r2_score(records_real, records_predict))
        plt.plot(records_predict.tolist(), 'm')
        plt.plot(records_real.tolist(), 'k')
        plt.show()
    else:
        print("fxxk")
    # 设置全局变量list2
    # gol.set_value('list2', list2)
    # xx = gol.set_value('list2', list2)
    # print(xx)





