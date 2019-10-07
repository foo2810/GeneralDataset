#

# Note
"""
GeneralDasetとの接続やデータの伝播周りがかなり複雑になってしまったので整理する必要あり
多分Chainを種類を一意に識別するidが必要 <- 複雑さを軽減
GeneralDatasetのinstance_rangeがどうなっているかまだ未確認

One-hotベクトルへの対応方法を検討
ラベルに対するfilter処理の検討
ラベルの分岐を許容するか

BufferedGetter内の_bufferは固定長である方がインデックス計算で都合がいい
ただし_bufferのサイズをデータフローの出力サイズではなく，初期化時に決める方がいい(なるべく大きめに)
"""

import numpy as np

from general_dataset.base import GeneralDatasetChainMixin

class _NodeChain(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return [x]

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
        self._buffer = self.gdataset.forward(0)
        self._pre_idx = 0
        self._suf_idx = len(self._buffer[0]) - 1
        self._cur_root_idx = 0
        self.out_length = len(self._buffer[0])
    
    def _area_check(self, idx):
        if self._pre_idx <= idx and idx <= self._suf_idx:
            return True
        else:
            return False
    
    def _update_buffer(self, idx, size=None):
        new_root_idx = idx // self.out_length
        self._buffer = self.gdataset.forward(new_root_idx)
        self._suf_idx = (new_root_idx + 1) * self.out_length - 1
        self._pre_idx = new_root_idx * self.out_length
        self._cur_root_idx = new_root_idx

    def calc_n_instances(self):
        # root chainを探索
        itr = self.gdataset.in_chain
        while type(itr) == GeneralDataset:
            itr = itr.in_chain

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

# kerasでいうModelクラスと同等
# GeneralDatasetChainMixinを前提
class GeneralDataset(GeneralDatasetChainMixin):
    def __init__(self, in_chain, out_chain=None):
        super().__init__()

        self.in_chain = in_chain
        if out_chain is None:
            tmp = _NodeChain()(in_chain)
            self.out_chain = tmp
        else:
            self.out_chain = out_chain
        
        self._buffered_getter = _BufferedGetter(self)
        
    def __call__(self, chain):
        _ = super().__call__(chain)

        # GeneralDatasetとChain(GeneralDatasetを含む)を接続するとき
        # データの伝播のためにselfの先頭chainとself.prevの末尾chainを接続する必要がある
        # カプセル化を徹底したいなら_NodeChainをGeneralDatasetの端点に用意すればできるがとりあえずはやってない
        chain.next = self.in_chain
        self.in_chain.prev = chain

        return self
    
    def compile(self):
        self.n_instances = self._buffered_getter.calc_n_instances()
        return self
    
    def forward(self, root_idx):
        # GeneralDatasetがRootChainのみで構成されている場合
        if self.in_chain is self.out_chain:
            return self.in_chain[root_idx]
        
        data = self.in_chain[root_idx]
        itr = self.in_chain
        while itr.next is not None:
            itr = itr.next
            data = itr.forward(data)
        return data

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
    
    def __getitem__(self, idx):
        return self._buffered_getter[idx]

    def __len__(self):
        # compile時に要素数を計算し保持
        return self.n_instances
    
    """
    def __iter__(self):
        for itr in self.out_chain:
            yield itr
    """
