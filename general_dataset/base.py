# 

import numpy as np

class GeneralDatasetChainMixin:
    def __init__(self):
        self.prev = None
        self.next = None
        self.instance_range = None

    def __call__(self, chain):
        chain.next = self
        self.prev = chain
        return self
    
    """ 結局初期化を担うので不要かもしれない """
    def load_data(self):
        pass

    # root chain以外のchainで実装が必要
    # iterableなValueを返す
    def filter(self, x):
        raise NotImplementedError

    def __getitem__(self, idx):
        if type(idx) == slice:
            ...
        else:
            prev_idx = idx // self.instance_range
            prev_out = self.prev[prev_idx]

            """ ラベルがない場合はどうするか？ """
            prev_data, prev_label = prev_out

            out_buffer = self.filter(prev_data)
            return out_buffer[idx % self.instance_range], prev_label

    def __len__(self):
        return len(self.prev) * self.instance_range

    def __iter__(self):
        if self.prev is None:
            # root chainのみで実装が必要
            raise NotImplementedError
        else:
            for prev_itr in self.prev:
                prev_data, prev_label = prev_itr
                for data in self.filter(prev_data):
                    yield data, prev_label


from keras.utils.data_utils import Sequence
class KerasDatasetBase(Sequence):
    def __init__(self, gdataset, batch_size=32):
        self.gdataset = gdataset
        self.batch_size = batch_size
    
    @property
    def nfiles(self):
        return len(self.gdataset)
    
    def __getitem__(self, idx):
        X = []
        labels = []
        for i in range(self.batch_size):
            x, l = self.gdataset[i+self.batch_size*idx]
            X.append(x)
            labels.append(l)
        return np.array(X), np.array(labels)
    
    def __len__(self):
        return self.nfiles // self.batch_size
