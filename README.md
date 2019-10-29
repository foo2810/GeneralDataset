# GeneralDataset

## ***What's GeneralDataset?***

Under Construction...

---

## ***How to use?***

GeneralDatasetではChainerの計算グラフのように各chainでデータの前処理を行いながらデータを流す．こうすることで，前処理における"車輪の再開発"を防ぐことができる上に，パーツを組み合わせるようにデータセットを構築することができる．また，General(汎用的な)Dataset(データセット)の名の通り，各機械学習ライブラリに対応したデータセットに変換することも可能である．以下では，general_datasetパッケージの基本的な使い方を示す．

`general_dataset/image_dataet.py`はこのパッケージの利用例にもなっているので，具体例はそちらを参照．

### **Step. 1: Create Root**

Rootは機械学習モデルでいうところのInputである．したがって，Rootは基本的に生データを出力することを想定している．

```
from general_dataset.mixin import GeneralDatasetRootMixin

class MyRoot(GeneralDatasetRootMixin):
    def __init__(self, ...):
        super().__init__()  # 必須

        # 出力するデータの準備等
    
    def __getitem__(self, idx):
        # Need to implement!!
        # ここで返す値は(data, label)の形である必要がある．
        # dataは基本的にnumpy.ndarrayで，shapeは(インスタンス数, 1インスタンスのデータのshape)
        # labelも基本的にnumpy.ndarrayで，shapeは(インスタンス数, labelのshape)
    
    def __len__(self):
        # Need to implement!!
        # 総インスタンス数を返すように実装
    
    @property
    def nclasses(self):
        # まだ未対応のメソッド
        # 実装しておくといいことがあるかもしれない
        # クラス数を返すように実装
```

### **Step. 2: Create Chain**

ChainではRootから出力される(生)データに対して階層的に前処理を行う．Chainは別のChainを組み合わせて構成することも可能である．

```
from general_dataset.mixin import GeneralDatasetChainMixin

class MyChainSub1(GeneralDatasetChainMixin):
    def __init__(self, ...):
        super().__init__()  # 必須
    
    def forward(self, x_y):
        # Need to implement!!
        # 何らかの前処理(特徴量選択, ...etc)

class MyChainSub2(GeneralDatasetChainMixin):
    def __init__(self, ...):
        super().__init__()  # 必須

    def forward(self, x_y):
        # Need to implement!!
        # 何らかの前処理(特徴量選択, ...etc)


class MyChainMain(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()  # 必須

        self.chain1 = MyChainSub1()
        self.chain2 = MyChainSub2()
    
    def forward(self, x_y):
        x_y = self.chain1(x_y)
        x_y = self.chain2(x_y)
        return x_y
```

### **Step. 3: Create General Dataset**

ここまでで作成した`Root`と`Chain`をまとめて`GeneralDataset`をつくる．

```
from general_dataset.dataset import GeneralDataset

root = MyRoot()
chain = MyCahinMain()

general_dataset = GeneralDataset(root, chain)
```

Complete!!