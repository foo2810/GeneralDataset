# Keras Dataset Base

import numpy as np

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
