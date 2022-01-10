# -*- coding:utf-8 -*-

import abc
from random import sample
from math import exp, log
import numpy as np
from model.tree import construct_decision_tree
from model.configs import Configs
from model.dataset import DataSet
from model.loss import MutiLeastSquaresError, MSE


class GBDT_Muti:
    def __init__(self, configs: Configs):
        self.configs = configs
        self.max_iter = configs.max_iter
        self.sample_rate = configs.sample_rate
        self.learn_rate = configs.learn_rate
        self.max_depth = configs.max_depth
        self.loss_type = configs.loss_type
        self.loss = None
        self.trees = dict()
        self.stop_iter = 0

    def fit(self, dataset: DataSet, valid_dataset: DataSet = None):

        if self.loss_type == "mse":
            self.loss = MutiLeastSquaresError(dataset.out_dim)
        else:
            raise ValueError(" Error loss function!")

        f = self.loss.initialize_f(dataset)  # f_0  [N*L]@ndarray

        best_loss = 100000
        for iter in range(1, self.max_iter + 1):
            idset = dataset.get_instances_idset()
            if 0 < self.sample_rate < 1:
                idset = sample(idset, int(len(idset) * self.sample_rate))  # N --> n
            train_data = dataset.get_sub_dataset(idset)  # ([n*D], [n*L])
            # 用损失函数的负梯度作为回归问题提升树的残差近似值
            residual = self.loss.compute_residual(dataset, idset, f)  # [n*L]
            leaf_nodes = []
            targets = residual[idset]
            tree = construct_decision_tree(
                train_data, targets, 0, leaf_nodes, self.configs, self.loss
            )
            # 验证损失和 early stop
            valid_f = self.compute_set_f_value(valid_dataset.X)
            valid_loss = self.compute_loss(valid_dataset, valid_f)
            print("iter%d : valid loss=%f" % (iter, valid_loss))
            if valid_loss < best_loss:
                best_loss = valid_loss
            else:
                print("Early stop at iter %d" % iter)
                self.stop_iter = iter
                break
            self.trees[iter] = tree
            self.loss.update_f_value(
                f, tree, leaf_nodes, idset, dataset, self.learn_rate
            )

            train_loss = self.compute_loss(dataset, f)
            print("iter%d : train loss=%f" % (iter, train_loss))

    def compute_loss(self, dataset, f):
        assert dataset.size() == len(f)
        loss = MSE(dataset.Y, f)
        return loss

    def compute_instance_f_value(self, instance):
        """计算样本的f值"""

        f_value = np.array([0.0] * self.loss.K)
        for iter in self.trees:
            f_value += self.learn_rate * self.trees[iter].get_predict_value(instance)

        return f_value

    def predict(self, instance):
        """
        返回f值
        """
        return self.compute_instance_f_value(instance)

    def predict_prob(self, instance):
        """返回属于每个类别的概率"""
        return self.predict(instance)

    def predict_label(self, instance):
        """预测标签"""
        f_i = self.compute_instance_f_value(instance)  # [L]@ndarray
        predict_label = np.zeros_like(f_i, dtype=int)
        predict_label[np.where(f_i[:] > self.configs.classify_shrd)] = 1
        return predict_label

    def compute_set_f_value(self, test_X):
        """
        test_X: [N*D]@ndarray
        """
        f_list = []
        for i in range(len(test_X)):
            f_i = self.compute_instance_f_value(test_X[i])
            f_list.append(f_i)
        f = np.array(f_list)
        return f

    def predict_set(self, test_X):
        """对测试集预测"""
        return self.compute_set_f_value(test_X)

    def predict_set_prob(self, test_X):
        """对测试集预测"""
        return self.compute_set_f_value(test_X)

    def predict_set_label(self, test_X):
        f = self.compute_set_f_value(test_X)
        predict_label = np.zeros_like(f, dtype=int)
        predict_label[np.where(f > self.configs.classify_shrd)] = 1
        return predict_label
