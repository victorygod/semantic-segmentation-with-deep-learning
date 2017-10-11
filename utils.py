import numpy as np
import os

label = [[64, 128, 64],	#Animal
[192, 0, 128],	#Archway
[0, 128, 192],	#Bicyclist
[0, 128, 64],	#Bridge
[128, 0, 0],	#Building
[64, 0, 128],	#Car
[64, 0, 192],	#CartLuggagePram
[192, 128, 64],	#Child
[192, 192, 128],	#Column_Pole
[64, 64, 128],	#Fence
[128, 0, 192],	#LaneMkgsDriv
[192, 0, 64],	#LaneMkgsNonDriv
[128, 128, 64],	#Misc_Text
[192, 0, 192],	#MotorcycleScooter
[128, 64, 64],	#OtherMoving
[64, 192, 128],	#ParkingBlock
[64, 64, 0],		#Pedestrian
[128, 64, 128],	#Road
[128, 128, 192],	#RoadShoulder
[0, 0, 192],		#Sidewalk
[192, 128, 128],	#SignSymbol
[128, 128, 128],	#Sky
[64, 128, 192],	#SUVPickupTruck
[0, 0, 64],		#TrafficCone
[0, 64, 64],		#TrafficLight
[192, 64, 128],	#Train
[128, 128, 0],	#Tree
[192, 128, 192],	#Truck_Bus
[64, 0, 64],		#Tunnel
[192, 192, 0],	#VegetationMisc
[0, 0, 0],		#Void
[64, 192, 0]]	#Wall

NUM_CLASS = len(label)

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
	for i in range(NUM_CLASS):
		h[label[i][0], label[i][1], label[i][2]] = i
	return h

def class_balancing():
	label_dir = "./label_camvid/"
	data_list = os.listdir(label_dir)
	l = len(data_list)
	sumLabel = np.zeros((NUM_CLASS))
	for now in range(l):
		label = np.load(label_dir+ data_list[now])
		shape = np.shape(label)
		label = np.reshape(label, (shape[0]*shape[1], NUM_CLASS))
		sumLabel += np.sum(label, axis = 0)/10000
		print(now)
		print(sumLabel)

	sumLabel = l/(sumLabel+1e-8)
	norm = np.sum(sumLabel)
	sumLabel = sumLabel / norm
	return sumLabel

#set_npy_num()
#print(class_balancing())

# file_path = "./net_weights.npy"
# vgg = np.load(file_path, encoding='latin1').item()
# print(np.shape(vgg["conv2_2"][0]))