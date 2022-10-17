import argparse
import tkinter
import numpy as np
import pyGRNN.Grnn
from pyACO.ACO_ANT import Aco


class AcoGrnn:
    # def __init__(self, data, num_sample, step=20):
    def __init__(self, argrs):
        assert argrs.step >= 10, "Wrong step!"
        self.max_iter = argrs.max_iter
        self.best_error = 0
        self.samples = argrs.samples
        self.step = argrs.step
        self.aval_sigma_mat = []  # Available sigma matrix
        # 加载GRNN网络
        self.grnn = pyGRNN.Grnn.Grnn(argrs.data_path, self.samples, 1)
        self.grnn.prepare_data()
        self.create_sigma_mat(argrs.min_sigma, argrs.max_sigma, argrs.step)

    def create_sigma_mat(self, min_sigma, max_sigma, step):
        """
        创建可用sigma矩阵
        sigma设置光滑因子的部分
        :param min_sigma: 可用sigma矩阵下限
        :param max_sigma: 可用sigma矩阵上限
        :param step: 设置的步长
        """
        sigma_mat = np.linspace(min_sigma, max_sigma, step,
                                dtype=np.float).transpose()
        self.aval_sigma_mat = sigma_mat.tolist()


if __name__ == "__main__":
    """
    step设置sigma可行解的个数, samples设置训练样本的个数 
    切记！当你更改以上两个值时，同时也应在ACO_ANT.py修改(city_num, ant_num)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='pyGRNN/sine.txt', help="data path")
    parser.add_argument("--max_iter", type=int, default=50, help="maximum iteration number")
    parser.add_argument("--step", type=int, default=20, help="step of creating available sigma mat")
    parser.add_argument("--samples", type=int, default=190, help="if you have 100 samples, please set it as 90, "
                                                                 "other 10 samples as test dataset")
    parser.add_argument("--save", type=bool, default=True, help="if you want to save final sigma matrix")
    parser.add_argument("--min_sigma", type=float, default=0.01, help="minimum available sigma value")
    parser.add_argument("--max_sigma", type=float, default=0.2, help="maximum available sigma value")
    parser.add_argument("--alpha", type=float, default=2.0, help="information-inspired factors")
    parser.add_argument("--beta", type=float, default=1.0, help="heuristic factor strength")
    parser.add_argument("--rho", type=float, default=0.5, help="volatile factor")
    parser.add_argument("--q", type=float, default=2.0, help="information intensity")
    parser.add_argument("--edt", type=float, default=0.6, help="natural selection thresholds, it will decrease "
                                                               "pheromone randomly when the possibility of next city"
                                                               "exceeds this threshold")
    opt = parser.parse_args()

    aco_grnn = AcoGrnn(opt)
    print("ACO-GRNN Iteration Begin")
    # Create ACO networks
    aco = Aco(opt, tkinter.Tk(), aco_grnn)
    # Run ACO
    aco.mainloop()
    print("ACO-GRNN Iteration End")
    # Run GRNN using optimised sigma matrix
    aco_grnn.grnn.run(aco.final_sigma, False)
    print(u"优化、计算完成，程序退出...")
