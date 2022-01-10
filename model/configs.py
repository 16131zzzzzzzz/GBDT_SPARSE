# -*- coding:utf-8 -*-
import pandas as pd

# Set hyper parameters here


class Configs:
    """
    Args:
    max_depth:      最大树深 (int)
    min_points:     结点最少样本数 (int)
    lambd:          L2正则化系数 (float)
    sparse_k:       稀疏度限制k (int)
    learn_rate:     学习率 (float)
    sample_rate:    每个弱学习器使用样本比例 (float)[0.6,0.9]
    max_iter:       最大迭代次数(弱学习器个数) (int)
    early_stop_n:   提前停止询问次数 (int)
    loss_type:      损失函数 from {'mse'} (str)
    classify_thrd:  分类阈值 (float)(0, 1)
    """

    def __init__(
        self, filename,
    ):
        file_config = pd.read_json(filename)
        self.max_depth = int(file_config["max_depth"])
        self.min_points = int(file_config["min_points"])
        self.lambd = int(file_config["lambd"])
        self.sparse_k = int(file_config["sparse_k"])
        self.learn_rate = float(file_config["learn_rate"])
        self.sample_rate = float(file_config["sample_rate"])
        self.max_iter = int(file_config["max_iter"])
        self.early_stop_n = int(file_config["early_stop_n"])
        self.classify_shrd = float(file_config["classify_shrd"])
        self.loss_type = "mse"

    def describe(self):
        return self.__dict__

    def save_configs(self, filename):

        print("--")
