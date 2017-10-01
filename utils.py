import numpy as np
import os

label = [[0,0,0], 
		[128, 0, 0], 
		[0, 128, 0], 
		[128, 128, 0], 
		[0, 0, 128], 
		[128, 0, 128],
		[0, 128, 128], 
		[128, 128, 128], 
		[64, 0, 0], 
		[192, 0, 0], 
		[64, 128, 0], 
		[192, 128, 0], 
		[64, 0, 128], 
		[192, 0, 128], 
		[64, 128, 128], 
		[192, 128, 128], 
		[0, 64, 0], 
		[128, 64, 0], 
		[0, 192, 0], 
		[128, 192, 0], 
		[0, 64, 128]]

label = np.array(label)

def change_label_to_img(label_data):
    shape = np.shape(label_data) 
    img_shape = [shape[1], shape[2], 3]
    ans_img = np.zeros(img_shape)

    for i in range(shape[1]):
        for j in range(shape[2]):
            k = np.argmax(label_data[0,i,j,:])
            ans_img[i,j,:] = label[k]
    return ans_img


def get_data_list(file_path):
    with open(file_path, 'rt') as f:
        s = f.read()
        data_list = s.split()
        return data_list

recovery_path = './net_weights.npy'

def get_weights():
	weights_data = {}
	if os.path.isfile(recovery_path):
	    weights_data = np.load(recovery_path).item()
	    print("Recover from " + recovery_path)
	else:
	    file_path = "./vgg16.npy"
	    vgg = np.load(file_path, encoding='latin1').item()
	    weights_data["conv1_1"] = vgg["conv1_1"]
	    weights_data["conv1_2"] = vgg["conv1_2"]
	    weights_data["conv2_1"] = vgg["conv2_1"]
	    weights_data["conv2_2"] = vgg["conv2_2"]
	    weights_data["conv3_1"] = vgg["conv3_1"]
	    weights_data["conv3_2"] = vgg["conv3_2"]
	    weights_data["conv3_3"] = vgg["conv3_2"]
	    weights_data["conv4_1"] = vgg["conv4_1"]
	    weights_data["conv4_2"] = vgg["conv4_2"]
	    weights_data["conv4_3"] = vgg["conv4_3"]
	    weights_data["conv5_1"] = vgg["conv5_1"]
	    weights_data["conv5_2"] = vgg["conv5_2"]
	    weights_data["conv5_3"] = vgg["conv5_3"]
	    weights_data["fc6"] = vgg["fc6"]
	    weights_data["fc7"] = vgg["fc7"]
	    print("Initialize with vgg-16.")
	return weights_data


#======================= set num=-1 ============================
def set_npy_num(num=-1):
	data = np.load("./net_weights.npy").item()
	data["num"] = -1
	np.save("./net_weights.npy", data)

#======================= format weights =======================
def format_weights_from_fcn32():
	data = np.load('weights.npy').item()
	data_2 = {}

	for k in data:
		if k=="score_fr":
			data_2["conv8"] = [np.transpose(data[k][0]), data[k][1]]
		elif k == "upscore":
			data_2["deconv32"] = [np.transpose(data[k][0])]
		else:
			data_2[k]=[np.transpose(data[k][0]), data[k][1]]

	np.save("net_weights.npy", data_2)

def get_color_to_label_hash():
	h = np.zeros((255,255,255))
	for i in range(21):
		h[label[i][0], label[i][1], label[i][2]] = i
	return h

set_npy_num()