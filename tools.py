import numpy as np
import cv2


labelColor=[
[64, 128, 64],
[192, 0, 128],
[0, 128, 192],
[0, 128, 64],
[128, 0, 0],
[64, 0, 128],
[64, 0, 192],
[192, 128, 64],
[192, 192, 128],
[64, 64, 128],
[128, 0, 192],
[192, 0, 64],
[128, 128, 64],
[192, 0, 192],
[128, 64, 64],
[64, 192, 128],
[64, 64, 0],
[128, 64, 128],
[128, 128, 192],
[0, 0, 192],
[192, 128, 128],
[128, 128, 128],
[64, 128, 192],
[0, 0, 64],
[0, 64, 64],
[192, 64, 128],
[128, 128, 0],
[192, 128, 192],
[64, 0, 64],
[192, 192, 0],
[0, 0, 0],
[64, 192, 0],
]

def label2img(label):
	sh=np.shape(label)
	img=np.zeros([sh[0],sh[1],3],np.uint8)
	ks=np.argmax(label,axis=2)
	for i in range(sh[0]):
		for j in range(sh[1]):
			img[i,j]=labelColor[ks[i,j]]
	return img

def showImage(img,winName='win0'):
	ans=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
	cv2.imshow(winName,img)

