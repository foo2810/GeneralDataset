#

# Note
"""
GeneralDasetとの接続やデータの伝播周りがかなり複雑になってしまったので整理する必要あり
多分Chainを種類を一意に識別するidが必要 <- 複雑さを軽減
GeneralDatasetのinstance_rangeがどうなっているかまだ未確認
"""

from general_dataset.base import GeneralDatasetChainMixin
from general_dataset.base import KerasDatasetBase

class _NodeChain(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()
    
    def filter(self, x):
        return [x]

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
        
        """
        self._in_node = _NodeChain()
        self._out_node = _NodeChain()

        self.in_chain(self._in_node)
        self._out_node(self.out_chain)
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
        itr = self.in_chain

        if type(itr) == GeneralDataset:
            itr.compile()

        if itr.next is None:
            itr.instance_range = 1
        else:
            itr = self.in_chain.next
            while True:
                """ itr.prev[0]がない場合の適切なエラー処理を実装する必要がある """
                """ labelがない場合はどうするか？ """

                # 場合によってはGeneralDatasetクラスの先頭Chainがroot_chainではない可能性がある
                # itr.prevがNone <=> GeneralDatasetの先頭Chainがroot_chain　であることを確認する必要あり
                if itr.prev is None:
                    # self.prevがNoneとなるのはroot_chainを先頭chainとして持つGeneralDatasetがselfである場合か？
                    if self.prev is None:
                        itr.instance_range = 1
                    else:
                        prev_data0 = self.prev[0][0]
                        itr.instance_range = len(itr.filter(prev_data0))
                else:
                    prev_data0 = itr.prev[0][0]
                    itr.instance_range = len(itr.filter(prev_data0))


                if type(itr) == GeneralDataset:
                    itr.compile()

                if itr.next is None: break
                else: itr = itr.next
        return self

    def apply(self, kind, batch_size):
        if kind == "keras":
            applied_dataset = KerasDatasetBase(self, batch_size=batch_size)
        elif kind == "pytorch":
            ...
        elif kind == "chainer":
            ...
        else:
            raise ValueError("{} is unknown kind.".format(kind))

        return applied_dataset
    
    def __getitem__(self, idx):
        print(type(self))
        return self.out_chain[idx]

    def __len__(self):
        return len(self.out_chain)
    
    def __iter__(self):
        for itr in self.out_chain:
            yield itr

class ChainList:
    def __init__(self, chain_list):
        if len(chain_list) == 0:
            raise ValueError("chain_list must have more than one chain.")

        self.chain_list = chain_list

        c = chain_list[0]
        for chain in chain_list[1:]:
            c = chain(c)
