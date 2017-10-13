import numpy as np
import matplotlib. pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import os
import math
import utils

camvid_train_dir = "../../CamVid/701_StillsRaw_full/"
camvid_label_dir = "../../CamVid/LabeledApproved_full/"
label_dir = "./label_small_size/"
train_dir = './train_small_size/'
data_list = os.listdir(camvid_label_dir)

h = utils.get_color_to_label_hash()
l = len(data_list)

size = (240, 320)
# now = 0
# img = mpimg.imread(os.path.join(camvid_label_dir, data_list[now]))*255
# img = misc.imresize(img.astype('uint8'), size, interp = "nearest")
# mpimg.imsave("test.jpg", img)

for now in range(l):
    img = mpimg.imread(os.path.join(camvid_label_dir, data_list[now])) * 255
    img = misc.imresize(img.astype('uint8'), size, interp = "nearest")
    shape = np.shape(img)
    classes = np.zeros((shape[0], shape[1], utils.NUM_CLASS))
    for i in range(shape[0]):
        for j in range(shape[1]):
            m = h[int(img[i,j,0]), int(img[i,j,1]), int(img[i,j,2])]
            classes[i,j,int(m)] = 1
    np.save(label_dir + data_list[now][0:len(data_list[now]) - 6] + ".png.npy", classes)
    print(data_list[now])

print("============")

data_list = os.listdir(camvid_train_dir)
l = len(data_list)
for now in range(l):
	img = mpimg.imread(os.path.join(camvid_train_dir, data_list[now]))*255
	img = misc.imresize(img.astype('uint8'), size)
	np.save(train_dir + data_list[now] + ".npy", img)
	print(data_list[now])