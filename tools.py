#-*-coding: utf-8-*-#
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

def scalingfun(inarray,(minb,maxb)):
#scale array from [minb,maxb] to [0,1]	
	return (inarray-minb)/(maxb-minb)

def descalingfun(inarray,(minb,maxb)):
#re-scale array to [minb,maxb]
	return inarray*(maxb-minb)+minb

def imgBound(imgbatch,p=0.7):
#imgbatch:[batch,h,w,c]; return [mean,bound]
	sh=np.shape(imgbatch)
	imgf=np.reshape(imgbatch,[-1])
	dur_abs=abs(imgf-np.mean(imgf))
	index=np.argsort(imgf)
	bound=dur_abs[index[int(sh[0]*p)]]
	mean=np.mean(imgf)
	maxb=np.amax(imgf)
	minb=np.amin(imgf)
	return [mean,bound,minb,maxb]

def useBound(imgbatch,(minb,maxb)):
	imgf=np.reshape(imgbatch,[-1])
	num=np.shape(imgf)
	c=0.0
	for i in range(num[0]):
		if imgf[i]>=minb and imgf[i]<=maxb:
			c+=1
	return c/num[0]


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
	print type(ans)
	print np.shape(ans)
	cv2.imshow(winName,img)

def subimg(imgs,subplotnum):
#imgs: [batch, height, width, channels]; subplotnum: [h,w]
	imgs[0]
	result=imgs[0]
	for i in range(subplotnum[0]):
		row=imgs[subplotnum[1]*i]
		for j in range(1,subplotnum[1]):
			dur=imgs[subplotnum[1]*i+j]
			row=cv2.hconcat((row,dur))
		if i==0:
			result=row
		else:
			result=cv2.vconcat((result,row))
	return result
