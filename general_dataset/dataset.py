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

from general_dataset.mixin import GeneralDatasetRootMixin, GeneralDatasetChainMixin, GeneralDatasetMixin

import numpy as np

class NodeChain(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()
    
    def forward(self, x_y):
        return x_y[0], x_y[1]

# kerasでいうModelクラスと同等
# GeneralDatasetChainMixinを前提
class GeneralDataset(GeneralDatasetMixin):
    def __init__(self, in_chain, out_chain):

        if not isinstance(in_chain, GeneralDatasetRootMixin) and not isinstance(in_chain, GeneralDatasetMixin):
            raise ValueError("in_chain is not root.")

        if not isinstance(out_chain, GeneralDatasetChainMixin):
            raise ValueError("out_chain is not chain")

        self.in_chain = in_chain
        self.out_chain = out_chain
        """
        self._buffered_getter = _BufferedGetter(self)
        self.n_instances = self._buffered_getter.calc_n_instances()
        """
        super().__init__(in_chain)
        
    # 廃止予定
    """
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
    """
    
    def get_item(self, root_idx):
        """
        # GeneralDatasetがRootChainのみで構成されている場合
        if self.in_chain is self.out_chain:
            return self.in_chain[root_idx]
        """
        
        data = self.in_chain.get_item(root_idx)
        """
        itr = self.in_chain
        while itr.next is not None:
            itr = itr.next
            data = itr.forward(data)
        return data
        """
        return self.out_chain(data)

    """
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
    """
    
    """
    def __getitem__(self, idx):
        return self._buffered_getter[idx]

    def __len__(self):
        # compile時に要素数を計算し保持
        return self.n_instances
    """
    
    """
    def __iter__(self):
        for itr in self.out_chain:
            yield itr
    """
