# -*- coding:utf-8 -*-

import numpy as np
from model.booster import GBDT_Muti
from model.dataset import DataSet
from model.configs import Configs

# 方便观察debug的测试数据
x = [
    [1.0, 3.0, 4.0, 2.0, 6.0],
    [3.0, 5.0, 0.0, 5.0, 3.0],
    [4.0, 0.0, 9.0, 7.0, 1.0],
    [5.0, 8.0, 9.0, 1.0, 1.0],
    [4.0, 9.0, 0.0, 5.0, 9.0],
    [9.0, 8.0, 5.0, 6.0, 4.0],
    [1.0, 5.0, 7.0, 2.0, 3.0],
    [3.0, 2.0, 0.0, 5.0, 6.0],
    [4.0, 1.0, 2.0, 5.0, 3.0],
    [5.0, 2.0, 9.0, 1.0, 5.0],
    [4.0, 3.0, 4.0, 3.0, 3.0],
]
y = [
    [0, 1, 1],
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 0],
]
x_a = np.array(x)
y_a = np.array(y)


if __name__ == '__main__':
    inp_dim, out_dim = 10, 5

    x_train = np.random.rand(10000, inp_dim)
    y_train = np.random.randint(0, 2, size=(10000, out_dim))
    x_valid = np.random.rand(1000, inp_dim)
    y_valid = np.random.randint(0, 2, size=(1000, out_dim))

    dataset_train = DataSet(x_train, y_train)
    dataset_test = DataSet(x_valid, y_valid)
    configs = Configs(min_points=20, sparse_k=3)

    # dataset_debug = DataSet(x_t, y_a)
    # configs_debug = Configs(min_points=1, sparse_k=2, max_depth=5, sample_rate=0.9)

    gbdt = GBDT_Muti(configs)

    gbdt.fit(dataset_train)
