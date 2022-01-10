# -*- coding:utf-8 -*-
from math import log
from random import sample
import numpy as np
import heapq
from model.configs import Configs
from model.dataset import DataSet


class Tree:
    def __init__(self):
        self.split_feature = None
        self.leftTree = None
        self.rightTree = None
        # 对于real value的条件为<=，对于类别值得条件为=
        # 将满足条件的放入左树
        self.real_value_feature = True
        self.conditionValue = None
        self.leafNode = None

    def get_predict_value(self, instance):
        if self.leafNode:  # 到达叶子节点
            return self.leafNode.get_predict_value()
        if self.split_feature is None:
            raise ValueError("the tree is null")
        if (
            self.real_value_feature
            and instance[self.split_feature] <= self.conditionValue
        ):
            return self.leftTree.get_predict_value(instance)
        elif (
            not self.real_value_feature
            and instance[self.split_feature] == self.conditionValue
        ):
            return self.leftTree.get_predict_value(instance)
        return self.rightTree.get_predict_value(instance)

    def describe(self, addtion_info=""):
        if not self.leftTree or not self.rightTree:
            return self.leafNode.describe()
        leftInfo = self.leftTree.describe()
        rightInfo = self.rightTree.describe()
        info = (
            addtion_info
            + "{split_feature:"
            + str(self.split_feature)
            + ",split_value:"
            + str(self.conditionValue)
            + "[left_tree:"
            + leftInfo
            + ",right_tree:"
            + rightInfo
            + "]}"
        )
        return info


class LeafNode:
    def __init__(self, dataIDset):
        self.dataIDset = dataIDset
        self.predictValue = None  # h [L]@ndarray

    def describe(self):
        return "{LeafNode:" + str(self.predictValue) + "}"

    def get_idset(self):
        return self.dataIDset

    def get_predict_value(self):
        return self.predictValue

    def update_predict_value(self, h):
        self.predictValue = h


def MSE(values):
    """
    均平方误差 mean square error
    """
    if len(values) < 2:
        return 0
    mean = sum(values) / float(len(values))
    error = 0.0
    for v in values:
        error += (mean - v) * (mean - v)
    return error


def get_sorted_top_k(array, top_k, axis=-1, reverse=True):
    """
    多维数组排序
    Args:
        array: 多维数组
        top_k: 取数
        axis: 轴维度
        reverse: 是否倒序 True=k最大值

    Returns:
        top_sorted_scores: 值
        top_sorted_indexes: 位置
    """
    if reverse:
        # argpartition分区排序，在给定轴上找到最小的值对应的idx，partition同理找对应的值
        # kth表示在前的较小值的个数，带来的问题是排序后的结果两个分区间是仍然是无序的
        # kth绝对值越小，分区排序效果越明显
        axis_length = array.shape[axis]
        partition_index = np.take(
            np.argpartition(array, kth=-top_k, axis=axis),
            range(axis_length - top_k, axis_length),
            axis,
        )
    else:
        partition_index = np.take(
            np.argpartition(array, kth=top_k, axis=axis), range(0, top_k), axis
        )
    top_scores = np.take_along_axis(array, partition_index, axis)
    # 分区后重新排序
    sorted_index = np.argsort(top_scores, axis=axis)
    if reverse:
        sorted_index = np.flip(sorted_index, axis=axis)
    top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
    top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
    return top_sorted_scores, top_sorted_indexes


def Sparse_split(dataset, targets, lambd, k, S=1):
    """
    dataset: DataSet Object has X and Y
    X: R^D data, [N*D]
    Y: {0,1}^L label, [N*L]
    targets: fit targets (resdual)
    sortedIDs: X's index set after sortting by every feature, [D*N]
    lambd: regularization parameter  (float)
    k: sparsity constrain    (int)
    S: samples test interval (gap number) (int>=1)
    """
    if S < 1:
        S = 1

    N = dataset.sample_num  # 数据量
    D = dataset.inp_dim  # 特征维度
    L = dataset.out_dim  # 标签总数

    # 按各特征对样本排序得到的下标集合 用来取样本
    sortedIDs = dataset.get_sortedset_index_by_all_feature()

    g = targets  # 对应论文表示

    f_best = 0.0
    feature_id = None
    threshold = None
    for j in range(D):  # 遍历每个特征
        # # give p  [L] array
        # for s in range(L):      # 每个标签计算稀疏程度 (不用循环用np)
        #     p_l[s] = 0.0
        #     p_r[s] = np.sum(g, axis=0)      #每列求和
        p_r = np.sum(g, axis=0, dtype=np.float64)
        p_l = np.zeros_like(p_r, dtype=np.float64)

        for i in range(0, N, S):  # 每个样本放到左树 (使用间隔S,放左边S个才计算一次得分)

            # 当前样本放到左树后 左右的各标签稀疏度的变化 (用np就不用循环s了)
            # 使用间隔S后,本次放入左树的是i-S+1 ~ i
            for idx in range(i - S + 1, i + 1):
                if idx >= 0:
                    p_l += g[sortedIDs[j, idx]]
                    p_r -= g[sortedIDs[j, idx]]

            l_topk_value, l_topk_idx = get_sorted_top_k(np.abs(p_l), k, axis=0)
            r_topk_value, r_topk_idx = get_sorted_top_k(np.abs(p_r), k, axis=0)

            # Q_l = heapq.nlargest(k, np.abs(p_l))
            # Q_r = heapq.nlargest(k, np.abs(p_r))
            Q_l = l_topk_idx.tolist()
            Q_r = r_topk_idx.tolist()

            p_l_square_sum = 0.0
            p_r_square_sum = 0.0
            for s in Q_l:
                p_l_square_sum += p_l[s] ** 2
            for s in Q_r:
                p_r_square_sum += p_r[s] ** 2

            f = -(p_l_square_sum / (i + lambd) + p_r_square_sum / (N - i + lambd))
            if f < f_best:
                f_best = f
                feature_id = j
                threshold = dataset.X[sortedIDs[j, i], j]
    return feature_id, threshold, l_topk_idx, r_topk_idx


def construct_decision_tree(
    dataset: DataSet, targets, depth, leaf_nodes, configs, loss, topk_idx=None
):
    if depth < configs.max_depth and dataset.size() > configs.min_points:
        feature_id, threshold, l_topk_idx, r_topk_idx = Sparse_split(
            dataset=dataset,
            targets=targets,
            lambd=configs.lambd,
            k=configs.sparse_k,
            S=dataset.size() // 20,
        )
        left_data, right_data = dataset.split_2subsets(feature_id, threshold)
        left_idx, right_idx = dataset.get_split_location(feature_id, threshold)

        tree = Tree()
        tree.split_feature = feature_id
        tree.conditionValue = threshold
        tree.leftTree = construct_decision_tree(
            left_data,
            targets[left_idx],
            depth + 1,
            leaf_nodes,
            configs,
            loss,
            l_topk_idx,
        )
        tree.rightTree = construct_decision_tree(
            right_data,
            targets[right_idx],
            depth + 1,
            leaf_nodes,
            configs,
            loss,
            r_topk_idx,
        )
        return tree
    else:  # 是叶子节点
        node = LeafNode(dataset.get_instances_idset())

        p = np.sum(targets, axis=0, dtype=np.float64)  # [L]
        h = np.zeros_like(p, dtype=np.float64)  # [L]
        h[topk_idx] = p[topk_idx] / (dataset.size() + configs.lambd)
        node.update_predict_value(h)

        leaf_nodes.append(node)
        tree = Tree()
        tree.leafNode = node
        return tree
