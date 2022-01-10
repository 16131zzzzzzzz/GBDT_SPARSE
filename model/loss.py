# -*- coding:utf-8 -*-

import abc
from math import exp, log
import numpy as np
from model.dataset import DataSet


def MSE(y, f):
    """
    y, f : [N*L]@ndarray
    """
    tmp = np.sum(np.square(y - f))
    return tmp / len(y)


class MutiLabelLossFunction(metaclass=abc.ABCMeta):
    def __init__(self, n_classes):
        self.K = n_classes

    @abc.abstractmethod
    def compute_residual(self, dataset, idset, f):
        """计算残差"""

    @abc.abstractmethod
    def update_f_value(
        self, f, tree, leaf_nodes, idset, dataset, learn_rate, label=None
    ):
        """更新F_{m-1}的值"""

    @abc.abstractmethod
    def initialize_f(self, f, dataset):
        """初始化F_{0}的值"""

    # @abc.abstractmethod
    # def update_ternimal_regions(self, targets, idset):
    #     """更新叶子节点的返回值"""


class MutiLeastSquaresError(MutiLabelLossFunction):
    """用于回归的最小平方误差损失函数"""

    def __init__(self, n_classes):
        super(MutiLeastSquaresError, self).__init__(n_classes)

    def compute_residual(self, dataset: DataSet, idset, f):
        y = dataset.get_instance_labels_set(idset)  # [n*L]@ndarray  n<N
        residual = np.zeros_like(f, dtype=np.float64)

        residual[idset] = y - f[idset]

        return residual

    def update_f_value(self, f, tree, leaf_nodes, idset, dataset: DataSet, learn_rate):
        data_idset = set(dataset.get_instances_idset())
        idset = set(idset)
        for node in leaf_nodes:
            for id in node.get_idset():
                f[id] += learn_rate * node.get_predict_value()
        for id in data_idset - idset:
            f[id] += learn_rate * tree.get_predict_value(dataset.get_instance(id))

    def initialize_f(self, dataset):
        """初始化F0，我们可以用训练样本的所有值的平均值来初始化，这里初始化为0.0"""
        f = np.zeros_like(dataset.Y, dtype=np.float64)
        return f
