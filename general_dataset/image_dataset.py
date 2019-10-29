# Image Dataset

from general_dataset.mixin import GeneralDatasetChainMixin, GeneralDatasetRootMixin, GeneralDatasetMixin

import numpy as np
from PIL import Image
from pathlib import Path

import sys
import tarfile
from urllib.request import urlretrieve


class STL10(GeneralDatasetRootMixin):
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

        labeled_images_fpath = self.save_dir / self.bin_name / "train_X.bin"
        labels_fpath = self.save_dir / self.bin_name / "train_y.bin"
        #unlabeled_images_fpath = self.save_dir / self.bin_name / "unlabeled_X.bin"

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
        #return self.labeled_images[idx], self.labels[idx]
        return self.labeled_images[idx][np.newaxis,...], self.labels[idx][np.newaxis,...]
    
    def __len__(self):
        return len(self.labeled_images)

"""
class STL10Dataset(GeneralDatasetMixin):
    def __init__(self, kind="train", save_dir="./"):
        self.kind = kind
        self.save_dir = save_dir

        self.stl10 = STL10(kind, save_dir)
        super().__init__(self.stl10)
    
    def get_item(self, root_idx):
        return self.stl10[root_idx]
"""
    

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
class ImagePathDataset(GeneralDatasetRootMixin):
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
        return np.array([self.instance_list[idx]]), np.array([self.label_list[idx]])
    
    def __len__(self):
        return self.n_instances
    
class ImageLoader(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()

    def forward(self, x_y):
        x, y = x_y
        tmp = []
        for xi in x:
            img = np.array(Image.open(xi))[np.newaxis,...]
            tmp.append(img)
        return np.concatenate(tmp), y

from general_dataset.preprocess import Rescale, Shift
class AugmentedImgChain1(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()
        self.preprocess_fn = np.vectorize(Rescale(1/255.))

    def forward(self, x_y):
        x, y = x_y
        return self.preprocess_fn(x), y

class AugmentedImgChain2(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()
        #self.preprocess_fn = np.vectorize(Shift(wShiftRange=0.5, hShiftRange=0.5))
        self.preprocess_fn = Shift(wShiftRange=0.5, hShiftRange=0.5)

    def forward(self, x_y):
        x, y = x_y
        
        if len(x) == 1:
            new_x = self.preprocess_fn(x[-1])[np.newaxis,...]
        elif len(x) > 1:
            tmp_x = []
            for xi, _ in zip(x, y):
                pimg = self.preprocess_fn(xi)[np.newaxis,...]
                tmp_x.append(pimg)

            new_x = np.concatenate(tmp_x)
        else:
            raise RuntimeError("len(x) <= 0 - AugmentedImgDataset2")

        return new_x, y

class ExpandImageChain(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()

    def forward(self, x_y):
        x, y = x_y
        tmp_x = []
        tmp_y = []
        for xi, yi in zip(x, y):
            xi = xi[np.newaxis,...]
            yi = yi[np.newaxis,...]
            tmp_x += [xi, xi]
            tmp_y += [yi, yi]
        return np.concatenate(tmp_x), np.concatenate(tmp_y)
