#-*-coding: utf-8-*-#
import tensorflow as tf
import numpy as np
import os
import random
import cv2
import tools

paramDict={}
paramBN={}
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad, some_other_arg):
  return gen_nn_ops._max_pool_grad(op.inputs[0],
                                   op.outputs[0],
                                   grad,
                                   op.get_attr("ksize"),
                                   op.get_attr("strides"),
                                   padding=op.get_attr("padding"),
                                   data_format='NHWC')


########[architecture]########
def getFilterGroup(shape=[7,7,64,64],name='filter0',trainable=True,init=None):
	with tf.name_scope(name) as scope:
		if init==None:
			kernel=tf.Variable(tf.truncated_normal(shape),trainable=True,name='kernel')
			bias=tf.Variable(tf.zeros(shape[-1]),trainable=True,name='bias')
		else:
			kernel=tf.Variable(tf.constant(init[0]),trainable=True,name='kernel')
			bias=tf.Variable(tf.constant(init[1]),trainable=True,name='bias')			
		fdict=[kernel,bias]
		#variableList[kernel.name]=kernel
		#variableList[bias.name]=bias
		return fdict

def deconvLayer(inTensor,inFilterShape,outputshape,activationFunc=tf.nn.relu,name="deconv0",trainable=True,init=None):
	#input are a tensor of shape [batch, h, w, channels] and a filter shape of [h, w, ch_out, ch_in]
	with tf.name_scope(name) as scope:
		filterDict=getFilterGroup(inFilterShape,"filter0",trainable=trainable,init=init)
		ans=tf.nn.conv2d_transpose(inTensor,filterDict[0],outputshape,strides=[1,1,1,1],padding="SAME",name="conv2d_transpose")
		paramDict[name]=filterDict
		return ans	
	
def convLayer(inTensor,inFilterShape,activationFunc=tf.nn.relu,name="conv0",trainable=True,init=None,BN=True):
	#input are a tensor of shape [batch, h, w, channels] and a filter shape of [h, w, ch_in, ch_out]
	with tf.name_scope(name) as scope:
		filterDict=getFilterGroup(inFilterShape,"filter0",trainable=trainable,init=init)
		ans=tf.nn.conv2d(inTensor,filterDict[0],strides=[1,1,1,1],padding="SAME",name="conv2d")
		ans_=ans+filterDict[1]
		if BN==True:
			#params that we actually use and put into trainable collection
			paramname={'moving_mean','moving_variance'}
			BNinit=[0,1]
			#If init is not none, ensure input init has 'init[2]'
			if init!= None and len(init)>2:
				BNinit=init[2]
			paramdict={}
			i=0
			for namedur in paramname:
				paramdict[namedur]=tf.constant_initializer(BNinit[i])
				i+=1
			with tf.variable_scope(name+"BN"):
				print(BNinit)
# ""is_training"" has a bug that we have to update params ourselves!!!! But this only fix the training phase problem
				ans__=tf.contrib.layers.batch_norm(ans_,center=False,scale=False,param_initializers=paramdict,is_training=True,trainable=True, decay=0.999)
			BNparamsCollection=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=(name+"BN"))
			for var in BNparamsCollection:
				print var.name
		paramBN[name+"BN"]=BNparamsCollection
		paramDict[name]=[filterDict[0],filterDict[1],BNparamsCollection]
		ans___=activationFunc(ans__)
		return ans___

def upsamplingLayer(indata, raveled_argmax, output_shape):
	input_shape =  tf.shape(indata)
	flat_input_size = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3]
	flat_output_shape = (output_shape[0], output_shape[1] * output_shape[2] * output_shape[3])
	values = tf.reshape(indata, [flat_input_size])
	batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=raveled_argmax.dtype),shape=(input_shape[0], 1, 1, 1))
	batch_num = tf.ones_like(raveled_argmax) * batch_range
	batch_num = tf.reshape(batch_num, (flat_input_size, 1))
	indices = tf.reshape(raveled_argmax, (flat_input_size, 1))
	indices = tf.concat(1,[batch_num, indices])
	output = tf.scatter_nd(indices, values, shape=tf.cast(flat_output_shape, tf.int64))
	output = tf.reshape(output, output_shape)
	return output

def memLayer(name='mem0',shape=[4,128,128,16], trainable = True,init=None):
	with tf.variable_scope(name) as scope:
		in_data = tf.zeros([shape[0], shape[1]*shape[2]*shape[3]])
		memory = tf.Variable(tf.truncated_normal([shape[1]*shape[2]*shape[3]]),trainable=trainable,name='memory')
		layer = tf.nn.bias_add(in_data, memory)
		layer = tf.reshape(layer, shape)
		return layer

def truncate(indata,(minb,maxb)):
#truncate a matrix by given bounds; indata should be a numpy array
	indata_=np.array(indata)
	maxl=indata_>maxb
	minl=indata_<minb
	return indata*(-maxl)*(-minl)+maxl*maxb+minl*minb

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

########[building and initializing]########

tf.reset_default_graph()

#we may change to bunch dim later#
#bunch=20
inc=3
outc=3
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
layers={}
indices={}

restoreMark=True
startPoint=9000
restorePath='./lightweighted/_iter_'+str(startPoint)+'.npy'


# nameMap insideParam:outsideParam
# As you change the model, remember to change the nameMap accordingly
'''
nameMap={'conv1':'conv1',
'conv2':'conv2',
'conv3':'conv3',
'mem3':'mem3',
'decoder3':'decoder3',
'mem2':'mem2',
'decoder2':'decoder2',
'mem1':'mem1',
'decoder1':'decoder1'
}
'''
nameMap={'conv1_1':'conv1_1',
'conv1_2':'conv1_2',
'conv2_1':'conv2_1',
'conv2_2':'conv2_2',
'conv3_1':'conv3_1',
'conv3_2':'conv3_2',
'decoder3_1':'decoder3_1',
'decoder3_2':'decoder3_2',
'decoder2_1':'decoder2_1',
'decoder2_2':'decoder2_2',
'decoder1_1':'decoder1_1',
'decoder1_2':'decoder1_2'
}

inits={}
if restoreMark:
	restoreFile=np.load(restorePath).item()
	for key in nameMap:
		if nameMap[key] in restoreFile.keys():
			inits[key]=restoreFile[nameMap[key]]
		else:
			inits[key]=None
else:
	for key in nameMap:
		inits[key]=None
	startPoint=1


####
memsize=4
batchsize=4
batchsizetf=tf.placeholder(tf.int32)
####

xd=tf.nn.dropout(x,keep_prob=0.5)
layers['conv1_1']=convLayer(xd,[7,7,3,16],name='conv1_1',trainable=True,init=inits['conv1_1'])
layers['conv1_2']=convLayer(layers['conv1_1'],[7,7,16,16],name='conv1_2',trainable=True,init=inits['conv1_2'])
[layers['pool1'],indices['pool1']]=tf.nn.max_pool_with_argmax(layers['conv1_2'], ksize=[1, 2, 2, 1], strides=[1, 1, 1,1],padding='SAME',name='pool1')


layers['conv2_1']=convLayer(layers['pool1'],[7,7,16,16],name='conv2_1',trainable=True,init=inits['conv2_1'])
layers['conv2_2']=convLayer(layers['conv2_1'],[7,7,16,16],name='conv2_2',trainable=True,init=inits['conv2_2'])
[layers['pool2'],indices['pool2']]=tf.nn.max_pool_with_argmax(layers['conv2_2'], ksize=[1, 2, 2, 1], strides=[1, 1, 1,1],padding='SAME',name='pool2')

layers['conv3_1']=convLayer(layers['pool2'],[7,7,16,16],name='conv3_1',trainable=True,init=inits['conv3_1'])
layers['conv3_2']=convLayer(layers['conv3_1'],[7,7,16,16],name='conv3_2',trainable=True,init=inits['conv3_2'])
[layers['pool3'],indices['pool3']]=tf.nn.max_pool_with_argmax(layers['conv3_2'], ksize=[1, 2, 2, 1], strides=[1, 1, 1,1],padding='SAME',name='pool3')

layers['uppool3']=upsamplingLayer(layers['pool3'],indices['pool3'],tf.shape(layers['conv3_2']))
layers['decoder3_1']=convLayer(layers['uppool3'],[7,7,16,16],name='decoder3_1',trainable=True,init=inits['decoder3_1'])
layers['decoder3_2']=convLayer(layers['decoder3_1'],[7,7,16,16],name='decoder3_2',trainable=True,init=inits['decoder3_2'])

layers['uppool2']=upsamplingLayer(layers['decoder3_2'],indices['pool2'],tf.shape(layers['conv2_2']))
layers['decoder2_1']=convLayer(layers['uppool2'],[7,7,16,16],name='decoder2_1',trainable=True,init=inits['decoder2_1'])
layers['decoder2_2']=convLayer(layers['decoder2_1'],[7,7,16,16],name='decoder2_2',trainable=True,init=inits['decoder2_2'])

layers['uppool1']=upsamplingLayer(layers['decoder2_2'],indices['pool1'],tf.shape(layers['conv1_2']))
layers['decoder1_1']=convLayer(layers['uppool1'],[7,7,16,16],name='decoder1_1',trainable=True,init=inits['decoder1_1'])
layers['decoder1_2']=convLayer(layers['decoder1_1'],[7,7,16,3],name='decoder1_2',trainable=True,init=inits['decoder1_2'])


loss0=tf.reduce_sum(tf.square(layers['decoder1_2']-y),axis=3)
loss0=tf.reduce_mean(loss0)
#L1=tf.reduce_sum(layers['pool2'],axis=3)
#L1=tf.reduce_mean(L1)
#L0=tf.count_nonzero(layers['pool2'])
#L0=tf.cast(L0,tf.float32)
#loss=loss0+0.01*L0
loss=loss0

lr=5e-4
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	trainStep=tf.train.RMSPropOptimizer(lr,decay=0.9).minimize(loss)
########[small utils]########
'''
for var in tf.global_variables():
	print var.name
	print var.get_shape()
'''

def showFeature(layer,imgshape=[8,8],name='feature-map',gamma=1):
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		com=sess.run(layer,feed_dict={x:[raws[0]],y:[labels[0]],batchsizetf:1})
	print imgBound(com,p=0.9)
	compimgsinputs=[]
	for i in range(imgshape[0]*imgshape[1]):
		compimgsinputs.append(com[0,:,:,i])
	compimgs=tools.subimg(compimgsinputs,imgshape)
	compimgs=truncate(compimgs,(0,1))
	compimgs=compimgs*255
	sh=np.shape(compimgs)
	compimgs_=np.zeros((sh[0],sh[1]),np.uint8)
	compimgs_=np.uint8(compimgs)
	cv2.imshow(name,compimgs_)

########[training]########

traindir='./data/'
testdir=''
nameList=os.listdir(traindir)
epoch=90000
test_iter=1000
save_iter=100



########[subtraction]########
#20150103_100047.npy day
#20150102_100105.npy day
#
testpath='./testgroup/day/'
testinstance='./testgroup/day/20150103_100047.npy'

with tf.Session() as sess:
	init=tf.global_variables_initializer()
	sess.run(init)
	print('init done')
	lossAvg=0
	lossNum=0
	for i in range(startPoint-1,epoch):
		names=random.sample(nameList,batchsize)
		raws=[]
		labels=[]
		for name in names:
			raw=np.load(traindir+name)
			raws.append(raw)
			label=np.load(traindir+name)
			labels.append(label)
		sess.run(trainStep,feed_dict={x:raws,y:labels,batchsizetf:batchsize})
		lossResult=sess.run(loss,feed_dict={x:raws,y:labels,batchsizetf:batchsize})
		lossAvg+=lossResult
		lossNum+=1
		print('iter_'+str(i+1)+': '+str(lossResult))
		if (i+1)%save_iter==0:
			print('average loss: %f'%(lossAvg/lossNum))
			param=sess.run(paramDict,feed_dict={x:raws,y:labels,batchsizetf:batchsize})
			np.save('./lightweighted/_iter_'+str(i+1)+'.npy',param)
			print('Model saved.')
		if (i+1)%test_iter==0:
#			for var in tf.trainable_variables():
#				print var.name
#				print var.get_shape()
			print('average loss: %f'%(lossAvg/lossNum))
			pred=sess.run(layers['decoder1_2'],feed_dict={x:[raws[0]],y:[labels[0]],batchsizetf:1})
			pred_=truncate(pred[0],(0,1))
			predshow=np.uint8(pred_*255/1.1)
			labelsshow=np.uint8(labels[0]*255)

			tools.showImage(predshow,'win1')
			tools.showImage(labelsshow,'win2')
			showFeature(layers['pool1'],[4,4],'pool1')
			showFeature(layers['pool2'],[4,4],'pool2')
#			showFeature(layers['pool3'],[4,4],'pool3')
#			showFeature(layers['pool4'],[4,4],'pool4')

			testinput=np.load(testinstance)
			testimg=sess.run(layers['decoder1_2'],feed_dict={x:[testinput],y:[testinput],batchsizetf:1})
			testimg_=truncate(testimg[0],(0,1))
			testshow=np.uint8(testimg_*255)
			tools.showImage(testshow,'test')

			cv2.waitKey()
			cv2.destroyAllWindows()
			lossAvg=0
			lossNum=0

print('OK')

