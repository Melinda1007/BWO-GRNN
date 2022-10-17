# -*- coding: utf-8 -*-
import copy
import random
import sys
import threading
import tkinter

import matplotlib.pyplot as plt
import numpy as np


# #  参数（信息启发因子alpha，启发因子强度beta，挥发系数rho，信息强度Q）
# (ALPHA, BETA, RHO, Q) = (2.0, 1.0, 0.5, 2.0)
# # 城市数，蚁群
# (city_num, ant_num) = (20, 190)
# # 自然选择阈值
# EDT = 0.6
#
# # 城市距离和信息素
# distance_graph = [ [1.0 for col in range(city_num)] for raw in range(city_num)]
# pheromone_graph = [ [1.0 for col in range(ant_num)] for raw in range(city_num)] # 50 x 20


class Ant(object):
    """定义蚂蚁"""

    # 初始化
    def __init__(self, ID, city_num, ants, alpha=None, beta=None, edt=None, available_sigma=None):

        self.ID = ID  # ID
        self.city_num = city_num
        self.ant_num = ants
        self.ALPHA = alpha
        self.BETA = beta
        self.EDT = edt
        self.sigma = available_sigma
        self.final_sigma = []
        self.error = sys.maxsize  # 当前蚂蚁获得的误差
        self.__clean_data()  # 随机初始化出生点

    # 初始数据
    def __clean_data(self):

        self.path = []  # 当前蚂蚁的路径           
        self.total_distance = 0.0  # 当前路径的总距离

        self.move_count = 0  # 移动次数
        self.current_city = -1  # 当前停留的城市
        self.open_table_city = [True for i in range(self.city_num)]  # 探索城市的状态

        city_index = random.randint(0, self.city_num - 1)  # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        # self.open_table_city[city_index] = False # 这里可以设置为True，当选择下一个城市时，可以选任何城市
        self.move_count = 1

    # 选择下一个城市
    def __choice_next_city(self, distance_graph, pheromone_graph):

        next_city = -1
        select_citys_prob = [0.0 for i in range(self.city_num)]
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in range(self.city_num):
            if self.open_table_city[i]:
                try:
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = (pow(pheromone_graph[self.current_city][i], self.ALPHA)
                                            * pow((1.0 / distance_graph[self.current_city][i]), self.BETA))
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError:
                    print(('Ant ID: {ID}, current city: {current}, target city: {target}'.
                           format(ID=self.ID, current=self.current_city, target=i)))
                    sys.exit(1)

        # 自然选择随机减少转移概率
        for i in range(self.city_num):
            pij = select_citys_prob[i] / total_prob
            if pij > self.EDT:
                rand_prob = random.uniform(0.0, 1.0)
                select_citys_prob[i] = select_citys_prob[i] * rand_prob

        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(self.city_num):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break

        # 未从概率产生，顺序选择一个未访问城市             
        if next_city == -1:
            for i in range(self.city_num):
                if self.open_table_city[i]:
                    next_city = i
                    break

        # 返回下一个城市序号
        return next_city

    # 计算路径总距离
    # 根据路径的距离判断最优路径（sigma的设定在这部分完成）
    # (这部分的设定是依据GRNN的错误率进行设定调节的，相当于适应度函数，即错误率最低的路径为最优路径)
    def __cal_total_distance(self, grnn=None):
        sigma = []
        for i in range(self.ant_num):
            sigma.append(self.sigma[self.path[i]])
        self.final_sigma = sigma
        self.error = grnn.run(sigma)

    def __move(self, next_city):
        """移动操作"""
        self.path.append(next_city)
        # self.open_table_city[next_city] = False # 进入下一个城市，每个城市依然可以选择
        self.current_city = next_city
        self.move_count += 1

    def search_path(self, grnn, dis_graph, phero_graph):
        """搜索路径"""
        # 初始化数据
        self.__clean_data()
        # 搜素路径，遍历完所有城市为止
        while self.move_count < self.ant_num:
            # 移动到下一个城市
            next_city = self.__choice_next_city(dis_graph, phero_graph)
            self.__move(next_city)

        # 计算路径总长度
        self.__cal_total_distance(grnn)

# 定义算法部分
class Aco(object):
    def __init__(self, argrs, root, aco_grnn):
        """
        初始化蚁群算法的参数
        :param root: 主函数的根
        :param ants: 蚂蚁个数，与样本数相等
        :param citys: 城市个数，与delta sigma的个数相同
        :param grnn: GRNN网络
        :param sigma_mat: 可行解矩阵，[1 x citys]
        :param max_iter: 最大迭代次数
        """
        self.root = root
        self.city_num = aco_grnn.step
        self.ant_num = aco_grnn.samples
        self.grnn = aco_grnn.grnn
        self.aval_sigma = aco_grnn.aval_sigma_mat
        self.max_iter = aco_grnn.max_iter
        self.Q = argrs.q
        self.RHO = argrs.rho
        self.ALPHA = argrs.alpha
        self.BETA = argrs.beta
        self.EDT = argrs.edt
        self.save = argrs.save
        self.distance_graph = [[1.0 for col in range(self.city_num)] for raw in range(self.city_num)]
        self.pheromone_graph = [[1.0 for col in range(self.ant_num)] for raw in range(self.city_num)]  # 50 x 20

        # 城市数目初始化为city_num
        self.best_sigma = []
        self.min_error = []
        self.final_sigma = []

        self.__r = 5
        self.__lock = threading.RLock()  # 线程锁

        self.__bindEvents()
        self.new()
        self.search_path()

    # 按键响应程序
    def __bindEvents(self):

        self.root.bind("q", self.quit)  # 退出程序
        self.root.bind("n", self.new)  # 初始化
        self.root.bind("e", self.search_path)  # 开始搜索
        self.root.bind("s", self.stop)  # 停止搜索

    # 更改标题
    def title(self, s):
        self.root.title(s)

    # 初始化
    def new(self):

        # 停止线程
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

        # 初始城市之间的距离和信息素
        for i in range(self.city_num):
            for j in range(self.ant_num):
                self.pheromone_graph[i][j] = 1.0

        # 初始蚁群，给定随机位置 random.randint(0, ant_num - 1)
        self.ants = [Ant(ID, self.city_num, self.ant_num, self.ALPHA, self.BETA, self.EDT,
                         self.aval_sigma) for ID in range(self.ant_num)]
        self.best_ant = Ant(-1, self.city_num, self.ant_num)  # 初始最优解
        self.best_ant.total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数

    def quit(self):
        """
        退出程序
        """
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.root.destroy()

    def stop(self, evt):
        """停止搜索"""
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

    def search_path(self, evt=None):
        """开始搜索"""
        self.__lock.acquire()  # 开启线程
        self.__running = True
        self.__lock.release()
        plt.ion()  # 开启交互模式
        while self.__running:
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径， 该蚂蚁获得一个误差
                ant.search_path(self.grnn, self.distance_graph, self.pheromone_graph)
                # 与当前最优蚂蚁比较
                if ant.error < self.best_ant.error:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            self.best_sigma.append(self.best_ant.final_sigma)
            self.min_error.append(self.best_ant.error)
            self.__update_pheromone_gragh()  # 更新信息素
            print(u"iteration number：", self.iter, u"minimum error：", self.best_ant.error)
            self.draw_chart()
            # 判断是否达到最大迭代
            if self.iter == self.max_iter:
                plt.savefig('error_chart.jpg', dpi=800)
                self.__running = False
                best_id = np.argmin(np.array(self.min_error))
                self.final_sigma = self.best_sigma[best_id]
                if self.save:
                    with open("sigma.txt", 'w') as wt:
                        for sigma in self.final_sigma:
                            wt.write('{:.6f}\n'.format(sigma))
                    wt.close()
                plt.ioff()  # 关闭交互
                self.quit()
            self.iter += 1

    def __update_pheromone_gragh(self):
        """更新信息素"""
        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(self.ant_num)] for raw in range(self.city_num)]
        for ant in self.ants:
            delta_pheromone = self.Q / ant.error
            for i in range(1, self.ant_num):
                start, end = ant.path[i - 1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与GRNN产生误差成反比
                temp_pheromone[start][end] += delta_pheromone
                # temp_pheromone[end][start] = temp_pheromone[start][end]

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(self.city_num):
            for j in range(self.ant_num):
                self.pheromone_graph[i][j] = self.pheromone_graph[i][j] * (1 - self.RHO) + temp_pheromone[i][j]

    def draw_chart(self, xlabel='iteration number', ylabel='error'):
        """实时更新误差曲线"""
        plt.clf()  # 刷新当前图表
        plt.title('Iteration Results')
        x = np.linspace(1, len(self.min_error), len(self.min_error))
        y = self.min_error
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y, color='red', linewidth=2.0)
        plt.fill_between(x, 0, y, facecolor='green', alpha=0.3)
        plt.pause(0.005)

    def mainloop(self):
        """主循环"""
        self.root.mainloop()


if __name__ == '__main__':
    Aco(tkinter.Tk()).mainloop()
    print('Done!')
