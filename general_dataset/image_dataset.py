# Image Dataset

from general_dataset.base import GeneralDatasetChainMixin

import numpy as np
from PIL import Image
from pathlib import Path

import sys
import tarfile
from urllib.request import urlretrieve


class STL10(GeneralDatasetChainMixin):
    def __init__(self, kind="train", save_dir="./"):
        super().__init__()

        self.DATA_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
        self.save_dir = Path(save_dir)
        self.bin_name = "stl10_binary"

        if kind in ("train", "test"):
            self.kind = kind
        else:
            # あまりよくない気もする
            raise ValueError("{} is invalid kind.")

        self.load_data()

    def load_data(self):
        self._download_and_extract()

        labeled_images_fpath = self.save_dir / "train_X.bin"
        labels_fpath = self.save_dir / "train_y.bin"
        #unlabeled_images_fpath = self.save_dir / "unlabeled_X.bin"

        with labeled_images_fpath.open("rb") as st:
            labeled_images = np.fromfile(st, dtype=np.uint8)
            labeled_images = labeled_images.reshape(-1, 3, 96, 96)
            labeled_images = np.transpose(labeled_images, (0, 3, 2, 1))
        
        with labels_fpath.open("rb") as st:
            labels = np.fromfile(st, dtype=np.uint8)
            labels = labels.reshape(-1, 1) - 1
        
        self.labeled_images = labeled_images
        self.labels = labels

    def _download_and_extract(self):
        if not self.save_dir.exists():
            raise FileNotFoundError("{} not found".format(self.save_dir))

        dl_fpath = self.save_dir / self.bin_name
        if not dl_fpath.exists():
            def __progress(count, block_size, total_size):
                sys.stdout.write('\rDownloading %s %.2f%%' % (str(dl_fpath), float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            
            fpath, _ = urlretrieve(self.DATA_URL, str(dl_fpath), reporthook=__progress)
            print("Downloaded {}".format(fpath))
            with tarfile.open(fpath, "r:gz") as tar:
                tar.extractall(path=str(self.save_dir))

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        return self.labeled_images[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.labeled_images)
    
    def __iter__(self):
        for i in range(len(self)):
            print("root")
            yield self.labeled_images[i], self.labels[i]


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
            raise IndexError(idx)
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