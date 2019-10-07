from general_dataset.image_dataset import ImagePathDatasetChain, ImageDatasetChain
from general_dataset.image_dataset import *
from general_dataset.image_dataset import AugmentedImgDataset1, AugmentedImgDataset2
from general_dataset.dataset import GeneralDataset

import matplotlib.pyplot as plt
from PIL import Image

from general_dataset.image_dataset import STL10

#stl10 = STL10(save_dir="./bin")
stl10 = STL10(save_dir=r"D:/Desktop/stl")
STL10Dataset = GeneralDataset(in_chain=stl10, out_chain=None).compile()

print("Num of STL10: {}".format(len(stl10)))
print("Num of STL10Dataset: {}".format(len(STL10Dataset)))

for i in range(5):
    print("NO.{}".format(i))
    img, label = STL10Dataset[i]
    pil_img = Image.fromarray(img)
    pil_img.save("{}.jpg".format(i))

keras_dataset = STL10Dataset.apply("keras", batch_size=32)
batch = keras_dataset[0]
imgs, labels = batch
for cnt, (img, label) in enumerate(zip(imgs, labels)):
    pil_img = Image.fromarray(img)
    pil_img.save("{}_keras.jpg".format(cnt))


dir_path = r"D:\OneDrive\Document\研究\絵師判定\train_data\artists"
#dir_path = r"C:\Users\kondo\OneDrive\Document\研究\絵師判定\train_data\artists"

img_path_dataset = ImagePathDatasetChain(dir_path, shuffle=False)
img_dataset = ImageDatasetChain()
augment = AugmentedImgDataset2()
expand = ExpandImageDataset()

dataset = img_dataset(img_path_dataset)
dataset2 = expand(augment)
#aug_img_dataset2 = AugmentedImgDataset2()(img_dataset)

base_img_dataset = GeneralDataset(in_chain=img_path_dataset, out_chain=dataset)
preprocess = GeneralDataset(in_chain=augment, out_chain=dataset2)

#dataset3 = preprocess(base_img_dataset)
dataset3 = preprocess(dataset)
#aug_img_dataset2 = augment(base_img_dataset)


#general_dataset = GeneralDataset(in_chain=img_path_dataset, out_chain=aug_img_dataset2)
#general_dataset = GeneralDataset(in_chain=base_img_dataset, out_chain=aug_img_dataset2)
#general_dataset = GeneralDataset(in_chain=base_img_dataset, out_chain=dataset3)
general_dataset = GeneralDataset(in_chain=img_path_dataset, out_chain=dataset3)
general_dataset.compile()

print("Num images: {}".format(len(general_dataset)))
#print("img_dataset - instance_range: {}".format(img_dataset.instance_range))
print("expand_dataset - instance_range: {}".format(expand.instance_range))

"""
for i in range(5):
    print("fname: {}".format(img_path_dataset[i]))
    img, _ = general_dataset[i]
    pil_img = Image.fromarray(img)
    pil_img.save("tmp_{}.png".format(i))
"""
