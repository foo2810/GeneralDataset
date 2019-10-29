from general_dataset.dataset import GeneralDataset
from general_dataset.mixin import GeneralDatasetChainMixin
from general_dataset.image_dataset import STL10

from general_dataset.image_dataset import *

import numpy as np
from PIL import Image

class MyDataset(GeneralDatasetMixin):
    def __init__(self, chain):
        super().__init__(STL10("train", save_dir=r"D:/Desktop/stl"), chain)
    
class MyChain(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()
        self.aug2 = AugmentedImgChain2()
        self.expand = ExpandImageChain()
    
    def forward(self, x_y):
        x_y = self.aug2(x_y)
        x_y = self.expand(x_y)
        return x_y

class MyChain3(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()
        self.expand = ExpandImageChain()
    
    def forward(self, x_y):
        return self.expand(x_y)

def test_stl():
    """
    stl10 = STL10Dataset(kind="train", save_dir=r"D:/Desktop/stl")
    my_chain = MyChain()
    stl10_dataset = GeneralDataset(in_chain=stl10, out_chain=my_chain)
    print("Num of STL10: {}".format(len(stl10)))
    print("Num of STL10Dataset: {}".format(len(stl10_dataset)))
    """
    mydataset = MyDataset(MyChain())
    stl10_dataset = GeneralDataset(in_chain=mydataset, out_chain=MyChain3())
    print("Num of STL10: {}".format(len(mydataset)))
    print("Num of STL10Dataset: {}".format(len(stl10_dataset)))
    
    for i in range(5):
        print("No.{}".format(i))
        img, label = stl10_dataset[i]
        img = np.uint8(img)
        #img = np.uint8(img * 255)
        pil_img = Image.fromarray(img)
        pil_img.save("{}.jpg".format(i))

    stl10_keras_ds = stl10_dataset.apply(kind="keras", batch_size=32)
    for batch in stl10_keras_ds:
        imgs, labels = batch
        print(imgs.shape)
        print(labels.shape)

class MyDataset2(GeneralDatasetMixin):
    def __init__(self, root, chain):
        super().__init__(root, chain)
        
class OriginalChain(GeneralDatasetChainMixin):
    def __init__(self):
        super().__init__()
        self.img_loader = ImageLoader()
        self.aug2 = AugmentedImgChain2()
        self.expand = ExpandImageChain()
    
    def forward(self, x_y):
        x_y = self.img_loader(x_y)
        x_y = self.aug2(x_y)
        x_y = self.expand(x_y)
        return x_y


def test_path_base_ds():
    dir_path = r"D:\OneDrive\Document\研究\絵師判定\train_data\artists"

    img_path_dataset = ImagePathDataset(dir_path, shuffle=False)
    mydataset = MyDataset2(img_path_dataset, OriginalChain())

    print("Num of img_path_dataset: {}".format(len(img_path_dataset)))
    print("Num of gdataset: {}".format(len(mydataset)))

    for i in range(5):
        print("No.{}".format(i))
        img, label = mydataset[i]
        img = np.uint8(img)
        #img = np.uint8(img * 255)
        pil_img = Image.fromarray(img)
        pil_img.save("{}.png".format(i))

test_stl()
#test_path_base_ds()