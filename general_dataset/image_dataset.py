# Image Dataset

from general_dataset.base import GeneralDatasetChainMixin

import numpy as np
from PIL import Image
from pathlib import Path


# クラスごとにディレクトリで分けられている構造のデータセットからパスを出力するデータセットを作成
# shuffle = Falseの場合，出力順はクラス順になる．(A, A, ..., A, B, ..., B, ..., E)
"""
directory + - class_A + - img_A1
          |           | - img_A2
          |           :
          |           + - img_An
          | - class_B
          :
          | - class_C
          :
          | - class_D
          :
          + - class_E
"""
class ImagePathDatasetChain(GeneralDatasetChainMixin):
    def __init__(self, directory, shuffle=False):
        super().__init__()

        self.directory = directory
        self.shuffle = shuffle

        self.train_path_list = self.val_path_list = None
        self.n_instances = 0

        self.load_data()
    
    def load_data(self):
        self.instance_list = []
        self.label_list = []
        class_id = 0
        classes = []
        for c in Path(self.directory).glob("*"):
            classes.append(c.name)

            for img_path in c.glob("*"):
                self.instance_list.append(str(img_path))
                self.label_list.append(class_id)
            class_id += 1

        self.n_instances = len(self.instance_list)
        self.instance_list = np.array(self.instance_list)
        self.label_list = np.array(self.label_list)

        if self.shuffle:
            p = np.random.permutation(len(self.instance_list))
            self.instance_list = self.instance_list[p]
            self.label_list = self.label_list[p]

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_instances:
            raise IndexError
        return self.instance_list[idx], self.label_list[idx]
    
    def __len__(self):
        return self.n_instances
    
    def __iter__(self):
        for i in range(self.n_instances):
            print("root")
            yield self.instance_list[i], self.label_list[i]

class ImageDatasetChain(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()

    def filter(self, x):
        return [np.array(Image.open(x))]

from general_dataset.preprocess import Rescale, Shift
class AugmentedImgDataset1(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()

        self.preprocess_fn = Rescale(1/255.)

    def filter(self, x):
        return [self.preprocess_fn(x)]

class AugmentedImgDataset2(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()
        self.preprocess_fn = Shift(wShiftRange=0.5, hShiftRange=0.5)

    def filter(self, x):
        return [self.preprocess_fn(x)]

class ExpandImageDataset(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()

    def filter(self, x):
        return [x, x]