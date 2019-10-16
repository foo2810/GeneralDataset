
from general_dataset.base import GeneralDatasetChainBase

import numpy as np

class _BufferedGetter:
    def __init__(self, gdataset):
        self.gdataset = gdataset

        self._buffer = None
        self._pre_idx = None
        self._suf_idx = None
        #self._pre_root_idx = 0
        self._cur_root_idx = None   # root_idx for current _buffer
        self.out_length = None      # Length of one _buffer

        self._init()
    
    def _init(self):
        # _bufferの構造
        # _buffer = (data, label)
        # data.shape = (size, data shape)
        # label.shape = (size, label shape)
        self._buffer = self.gdataset.get_item(0)
        self._pre_idx = 0
        self._suf_idx = len(self._buffer[0]) - 1
        self._cur_root_idx = 0
        self.out_length = len(self._buffer[0])
    
    def _clear(self):
        del self._buffer
        self._buffer = None
        self._pre_idx = None
        self._suf_idx = None
        self._cur_root_idx = None
        self.out_length = None
   
    def _area_check(self, idx):
        if idx >= self._pre_idx and idx <= self._suf_idx:
            return True
        else:
            return False
    
    def _update_buffer(self, idx, size=None):
        new_root_idx = idx // self.out_length
        self._buffer = self.gdataset.get_item(new_root_idx)
        self._suf_idx = (new_root_idx + 1) * self.out_length - 1
        self._pre_idx = new_root_idx * self.out_length
        self._cur_root_idx = new_root_idx

    def calc_n_instances(self):
        # root chainを探索
        itr = self.gdataset.root
        while isinstance(itr, GeneralDatasetMixin):
            itr = itr.root

        n_root_instances = len(itr)
        
        return n_root_instances * self.out_length

    def __getitem__(self, idx):

        # sliceによるアクセスは未完成
        if type(idx) == slice:
            # stepには未対応
            s_pos = idx.start
            e_pos = idx.stop
            #step = idx.step
            if self._area_check(s_pos) and not self._area_check(e_pos):
                tmpx = []
                tmpy = []
                sidx = s_pos % self.out_length
                tmpx.append(self._buffer[0][sidx:].copy())
                tmpy.append(self._buffer[1][sidx:].copy())
                self._update_buffer(e_pos)
                eidx = e_pos % self.out_length
                tmpx.append(self._buffer[0][eidx])
                tmpy.append(self._buffer[1][eidx])
                return np.concatenate(tmpx), np.concatenate(tmpy)

            elif not self._area_check(s_pos) and self._area_check(e_pos):
                tmpx = []
                tmpy = []
                eidx = e_pos % self.out_length
                tmpx.append(self._buffer[0][eidx:].copy())
                tmpy.append(self._buffer[1][eidx:].copy())
                self._update_buffer(s_pos)
                sidx = s_pos % self.out_length
                tmpx.append(self._buffer[0][sidx])
                tmpy.append(self._buffer[1][sidx])

                # ここで間違っていると学習結果がおかしくなる
                tmpx.reverse()
                tmpy.reverse()
                return np.concatenate(tmpx), np.concatenate(tmpy)

            elif not self._area_check(s_pos) and not self._area_check(e_pos):
                self._update_buffer(s_pos)
                sidx = s_pos % self.out_length

                if self._area_check(e_pos):
                    eidx = e_pos % self.out_length
                    return self._buffer[sidx:eidx]

                tmpx = []
                tmpy = []
                
                tmpx.append(self._buffer[0][sidx:])
                tmpy.append(self._buffer[1][sidx:])

                self._update_buffer(e_pos)
                eidx = e_pos % self.out_length
                tmpx.append(self._buffer[0][:eidx])



            else:
                sidx = s_pos % self.out_length
                eidx = e_pos % self.out_length
                return self._buffer[0][sidx:eidx], self._buffer[1][sidx:eidx]
        else:
            if not self._area_check(idx):
                self._update_buffer(idx)
            cidx = idx % self.out_length
            #print(self._buffer[0].shape)
            #print(self._buffer[1].shape)
            return self._buffer[0][cidx], self._buffer[1][cidx]


class GeneralDatasetRootMixin(GeneralDatasetChainBase):
    def __init__(self):
        super().__init__("root_chain")

    """ 結局初期化を担うので不要かもしれない """
    def load_data(self):
        pass

    def get_item(self, root_idx):
        return self[root_idx]    

    def __call__(self, root_idx):
        return self[root_idx]
    
    # batchもどきを返すように実装
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    @property
    def nclasses(self):
        raise NotImplementedError

class GeneralDatasetChainMixin(GeneralDatasetChainBase):
    def __init__(self):
        super().__init__("normal_chain")

    def __call__(self, x_y):
        return self.forward(x_y)

    # batchもどき単位の処理を定義
    # x_y: batch
    # batch[0].shape => (batch_size, instance shape)
    # batch[1].shape => (batch_size, label)
    def forward(self, x_y):
        raise NotImplementedError

# _buffered_getterの初期化ではGeneralDatasetMixinのサブクラスのforwardを呼び出す
# そのためこれを継承したクラスはsuper().__init__は__init__の最後で呼ぶのがベター
class GeneralDatasetMixin(GeneralDatasetChainBase):
    def __init__(self, root, no_buffer=False):
        super().__init__("dataset")

        self.root = root    # rootはGeneralDatasetMixinのサブクラスかGeneralDatasetRoot
        # _buffered_getterを無効にするフラグメモリを節約したいときに使う(for expert)
        self.no_buffer = no_buffer
        self._buffered_getter = _BufferedGetter(self)
        self.n_instances = self._buffered_getter.calc_n_instances()

        if self.no_buffer:
            del self._buffered_getter
            self._buffered_getter = None
        
    def apply(self, kind, batch_size):
        if kind == "keras":
            from general_dataset.keras_base import KerasDatasetBase
            applied_dataset = KerasDatasetBase(self, batch_size=batch_size)
        elif kind == "pytorch":
            ...
        elif kind == "chainer":
            ...
        else:
            raise ValueError("{} is unknown kind.".format(kind))

        return applied_dataset
    
    def __call__(self, root_idx):
        return self.get_item(root_idx)

    # flow時の起点として発火
    def get_item(self, root_idx):
        raise NotImplementedError

    # 1 instanceを返す
    def __getitem__(self, idx):
        if self._buffered_getter is None:
            raise RuntimeError("This dataset is no-buffer mode.")

        return self._buffered_getter[idx]

    def __len__(self):
        # compile時に要素数を計算し保持
        return self.n_instances
    
    """
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    """

