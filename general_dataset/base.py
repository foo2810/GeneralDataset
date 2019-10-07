# 

import numpy as np

class GeneralDatasetChainMixin:
    def __init__(self):
        self.prev = None
        self.next = None

    def __call__(self, chain):
        chain.next = self
        self.prev = chain
        return self

class GeneralDatasetRoot(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()

    """ 結局初期化を担うので不要かもしれない """
    def load_data(self):
        pass

    # batchもどきを返すように実装
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    @property
    def nclasses(self):
        raise NotImplementedError

class GeneralDatasetChain(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()

    # batchもどき単位の処理を定義
    # x_y: batch
    # batch[0].shape => (batch_size, instance shape)
    # batch[1].shape => (batch_size, label)
    def forward(self, x_y):
        raise NotImplementedError

