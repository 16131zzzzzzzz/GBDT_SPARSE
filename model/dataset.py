# -*- coding:utf-8 -*-
import numpy as np
from copy import deepcopy


class DataSet:
    def __init__(self, data_x, data_y, data_id=None):
        """
        data_x:  [N*D]@ndarray
        data_y:  [N*L]@ndarray
        data_id: [N*1]@ndarray  give the sample id which from original dataset
        """
        assert len(data_x) == len(data_y)
        self.X = data_x
        self.Y = data_y
        self.inp_dim = np.size(data_x, 1)  # D
        self.out_dim = np.size(data_y, 1)  # L
        self.sample_num = len(data_x)  # N

        if data_id is None:
            self.IDs = np.array([i for i in range(len(data_x))]).reshape(
                (len(data_x), 1)
            )
        else:
            self.IDs = data_id
        # IDs to idset: IDs.ravel().tolist()

    def describe(self):
        info = "DataSet: \n"
        info += "  sample_num: %d\n" % self.sample_num
        info += "  inp_dim: %d\n" % self.inp_dim
        info += "  out_dim: %d\n" % self.out_dim
        print(info)

    def get_instances_idset(self):
        """ 获取样本的id集合 (list)"""
        idxs = self.IDs.ravel().tolist()
        return idxs

    def is_real_type_field(self, name):
        """判断特征类型是否是real type"""
        if name not in self.field_names:
            raise ValueError(" field name not in the dictionary of dataset")
        return len(self.field_type[name]) == 0

    def size(self):
        """ 返回样本个数 """
        return len(self.X)

    def get_instance(self, idx):
        """ 根据ID获取样本 x """
        if idx >= self.sample_num:
            raise ValueError("Id not in the instances dict of dataset")
        return self.X[idx]

    def get_instance_labels(self, idx):
        """ 根据ID获取样本 y """
        if idx >= self.sample_num:
            raise ValueError("Id not in the instances dict of dataset")
        return self.Y[idx]

    def get_instance_with_labels(self, idx):
        """ 根据ID获取样本和标签 x y """
        if idx >= self.sample_num:
            raise ValueError("Id not in the instances dict of dataset")
        return self.X[idx], self.Y[idx]

    def get_instance_set(self, IDset):
        """ 根据idset获得样本子集X' IDset可以是list或1D ndarray"""
        return self.X[IDset]

    def get_instance_labels_set(self, IDset):
        """ 根据idset获得样本子集Y' IDset可以是list或1D ndarray"""
        return self.Y[IDset]

    def get_instance_set_with_labels(self, IDset):
        """ 根据idset获得样本子集X' Y' IDset可以是list或1D ndarray"""
        return self.X[IDset], self.Y[IDset]

    def get_sub_dataset(self, IDset):
        """ 根据idset获得数据集子集D'=(X', Y') IDset可以是list或1D ndarray"""
        sub_data = DataSet(self.X[IDset], self.Y[IDset], self.IDs[IDset])
        return sub_data

    def get_sub_dataset_copy(self, IDset):
        """ 根据idset获得数据集子集D'=(X', Y') IDset可以是list或1D ndarray"""
        sub_data = DataSet(
            deepcopy(self.X[IDset]), deepcopy(self.Y[IDset]), deepcopy(self.IDs[IDset])
        )
        return sub_data

    def get_sortedset_index_by_feature(self, feature_id):
        # Get the index set of samples X in ascending order by a feature
        idset = self.X[:, feature_id].argsort().tolist()  # [N]@list
        return idset

    def get_sortedset_index_by_all_feature(self):
        # Get the index set of samples X in ascending order by each feature
        # Then a sample can be getted by
        # X[sortedIDs[sort_feature_id, ranked_sample_id], select_feature_id]
        sortedIDs = self.X.T.argsort()  # [D*N]@ndarray
        return sortedIDs

    def get_split_location(self, feature_id, threshold):
        left_idx = np.where(self.X[:, feature_id] <= threshold)
        right_idx = np.where(self.X[:, feature_id] > threshold)

        return left_idx, right_idx

    def split_2subsets(self, feature_id, threshold):
        left_idx, right_idx = self.get_split_location(feature_id, threshold)

        left_X = self.X[left_idx]
        left_Y = self.Y[left_idx]
        left_IDs = self.IDs[left_idx]
        left_dataset = DataSet(left_X, left_Y, left_IDs)

        right_X = self.X[right_idx]
        right_Y = self.Y[right_idx]
        right_IDs = self.IDs[right_idx]
        right_dataset = DataSet(right_X, right_Y, right_IDs)

        return left_dataset, right_dataset

    def split_2subIDs(self, feature_id, threshold):
        left_idx, right_idx = self.get_split_location(feature_id, threshold)

        left_IDs = self.IDs[left_idx]
        right_IDs = self.IDs[right_idx]

        return left_IDs, right_IDs

    def split_2subidsets(self, feature_id, threshold):
        left_IDs, right_IDs = self.split_2subIDs(feature_id, threshold)

        left_idset = left_IDs.ravel().tolist()
        right_idset = right_IDs.ravel().tolist()
        return left_idset, right_idset
