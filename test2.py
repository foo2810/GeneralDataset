from general_dataset.image_dataset import ImagePathDataset, ImageDataset
from general_dataset.image_dataset import AugmentedImgDataset1, AugmentedImgDataset2, ExpandImageDataset
from general_dataset.dataset import GeneralDataset
from general_dataset.image_dataset import STL10

import numpy as np
from PIL import Image

stl10 = STL10(save_dir=r"D:/Desktop/stl")
aug1 = AugmentedImgDataset1()
aug2 = AugmentedImgDataset2()
expand = ExpandImageDataset()

ds = stl10
ds = aug2(aug1(ds))

stl10_dataset = GeneralDataset(in_chain=stl10, out_chain=ds).compile()

print("Num of STL10: {}".format(len(stl10)))
print("Num of STL10Dataset: {}".format(len(stl10_dataset)))

for i in range(5):
    print("No.{}".format(i))
    img, label = stl10_dataset[i]
    img = np.uint8(img * 255)
    pil_img = Image.fromarray(img)
    pil_img.save("{}.jpg".format(i))

stl10_keras_ds = stl10_dataset.apply(kind="keras", batch_size=32)

for batch in stl10_keras_ds:
    imgs, labels = batch
    print(imgs.shape)
    print(labels.shape)