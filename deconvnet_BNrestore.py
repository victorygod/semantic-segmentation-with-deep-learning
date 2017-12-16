#-*-coding: utf-8-*-#
import tensorflow as tf
import numpy as np
import os
import random
import cv2
import tools

paramDict={}

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
	#input are a tensor of shape [batch, h, w, channels] and a filter shape of [h, w, ch_in, ch_out]
	with tf.name_scope(name) as scope:
		filterDict=getFilterGroup(inFilterShape,"filter0",trainable=trainable,init=init)
		ans=tf.nn.conv2d_transpose(inTensor,filterDict[0],outputshape,strides=[1,1,1,1],padding="SAME",name="conv2d_transpose")
		paramDict[name]=filterDict
		return ans	
		
def convLayer(inTensor,inFilterShape,activationFunc=tf.nn.relu,name="conv0",trainable=True,init=None):
	#input are a tensor of shape [batch, h, w, channels] and a filter shape of [h, w, ch_in, ch_out]
	with tf.name_scope(name) as scope:
		filterDict=getFilterGroup(inFilterShape,"filter0",trainable=trainable,init=init)
		ans=tf.nn.conv2d(inTensor,filterDict[0],strides=[1,1,1,1],padding="SAME",name="conv2d")
		ans=activationFunc(ans+filterDict[1])
		BNinit=[0,0,0,0]
		#ensure input init has 'init[2]'
		if init!= None:
			BNinit=init[2]
		with tf.variable_scope(name+"BN"):
			ans=tf.contrib.layers.batch_norm(ans,param_initializers={'beta':tf.constant_initializer(BNinit[0]),'gamma':tf.constant_initializer(BNinit[1]),'moving_mean':tf.constant_initializer(BNinit[2]),'moving_variance':tf.constant_initializer(BNinit[3])},is_training=True, decay=0.999)
		BNparamsCollection=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=(name+"BN"))
		paramDict[name]=[filterDict[0],filterDict[1],BNparamsCollection]
		return ans

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

restoreMark=False
startPoint=1000
restorePath='./models_4layer/_iter_'+str(startPoint)+'.npy'


# nameMap insideParam:outsideParam
# As you change the model, remember to change the nameMap accordingly
nameMap={'conv1':'conv1',
'conv2':'conv2',
'conv3':'conv3',
'conv4':'conv4',
'decoder4':'decoder4',
'decoder3':'decoder3',
'decoder2':'decoder2',
'decoder1':'decoder1'
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



layers['conv1']=convLayer(x,[3,3,3,16],name='conv1',trainable=False,init=inits['conv1'])
[layers['pool1'],indices['pool1']]=tf.nn.max_pool_with_argmax(layers['conv1'], ksize=[1, 2, 2, 1], strides=[1, 2, 2,1],padding='SAME',name='pool1')


layers['conv2']=convLayer(layers['pool1'],[3,3,16,16],name='conv2',trainable=True,init=inits['conv2'])
[layers['pool2'],indices['pool2']]=tf.nn.max_pool_with_argmax(layers['conv2'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2')


layers['conv3']=convLayer(layers['pool2'],[3,3,16,16],name='conv3',trainable=True,init=inits['conv3'])
[layers['pool3'],indices['pool3']]=tf.nn.max_pool_with_argmax(layers['conv3'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool3')

layers['conv4']=convLayer(layers['pool3'],[3,3,16,16],name='conv4',trainable=True,init=inits['conv4'])
[layers['pool4'],indices['pool4']]=tf.nn.max_pool_with_argmax(layers['conv4'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool4')
'''
layers['conv3_1']=convLayer(layers['pool2'],[3,3,128,256],name='conv3_1',trainable=False,init=inits['conv3_1'])
layers['conv3_2']=convLayer(layers['conv3_1'],[3,3,256,256],name='conv3_2',trainable=False,init=inits['conv3_2'])
layers['conv3_3']=convLayer(layers['conv3_2'],[3,3,256,256],name='conv3_3',trainable=False,init=inits['conv3_3'])
layers['pool3']=tf.nn.avg_pool(layers['conv3_3'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool3')

layers['conv4_1']=convLayer(layers['pool3'],[3,3,256,512],name='conv4_1',trainable=False,init=inits['conv4_1'])
layers['conv4_2']=convLayer(layers['conv4_1'],[3,3,512,512],name='conv4_2',trainable=False,init=inits['conv4_2'])
layers['conv4_3']=convLayer(layers['conv4_2'],[3,3,512,512],name='conv4_3',trainable=False,init=inits['conv4_3'])
layers['pool4']=tf.nn.avg_pool(layers['conv4_3'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool4')

layers['conv5_1']=convLayer(layers['pool4'],[3,3,512,512],name='conv5_1',trainable=False,init=inits['conv5_1'])
layers['conv5_2']=convLayer(layers['conv5_1'],[3,3,512,512],name='conv5_2',trainable=False,init=inits['conv5_2'])
layers['conv5_3']=convLayer(layers['conv5_2'],[3,3,512,512],name='conv5_3',trainable=False,init=inits['conv5_3'])
layers['pool5']=tf.nn.avg_pool(layers['conv5_3'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool5')

########

layers['uppool5']=upsamplingLayer(layers['pool5'],indices['pool5'],tf.shape(layers['conv5_3']))
layers['decoder5_1']=convLayer(layers['uppool5'],[3,3,512,256],name='decoder5_1',trainable=True,init=inits['decoder5_1'])
layers['decoder5_2']=convLayer(layers['decoder5_1'],[3,3,256,256],name='decoder5_2',trainable=True,init=inits['decoder5_2'])

layers['uppool4']=upsamplingLayer(layers['decoder5_2'],indices['pool4'],tf.shape(layers['conv4_3']))
layers['decoder4_1']=convLayer(layers['uppool4'],[3,3,512,512],name='decoder4_1',trainable=True,init=inits['decoder4_1'])
layers['decoder4_2']=convLayer(layers['decoder4_1'],[3,3,512,512],name='decoder4_2',trainable=True,init=inits['decoder4_2'])
layers['decoder4_3']=convLayer(layers['decoder4_2'],[3,3,512,512],name='decoder4_3',trainable=True,init=inits['decoder4_3'])

layers['uppool5']=upsamplingLayer(layers['pool5'],indices['pool5'],tf.shape(layers['conv5_3']))
layers['decoder5_1']=convLayer(layers['uppool5'],[3,3,512,512],name='decoder5_1',trainable=True,init=inits['decoder5_1'])
layers['decoder5_2']=convLayer(layers['decoder5_1'],[3,3,512,512],name='decoder5_2',trainable=True,init=inits['decoder5_2'])
'''
layers['uppool4']=upsamplingLayer(layers['pool4'],indices['pool4'],tf.shape(layers['conv4']))
layers['decoder4']=deconvLayer(layers['uppool4'],[3,3,16,16],tf.shape(layers['pool3']),name='decoder4',trainable=True,init=inits['decoder4'])

layers['uppool3']=upsamplingLayer(layers['decoder4'],indices['pool3'],tf.shape(layers['conv3']))
layers['decoder3']=deconvLayer(layers['uppool3'],[3,3,16,16],tf.shape(layers['pool2']),name='decoder3',trainable=True,init=inits['decoder3'])


layers['uppool2']=upsamplingLayer(layers['pool2'],indices['pool2'],tf.shape(layers['conv2']))
layers['decoder2']=deconvLayer(layers['uppool2'],[3,3,16,16],tf.shape(layers['pool1']),name='decoder2',trainable=True,init=inits['decoder2'])

layers['uppool1']=upsamplingLayer(layers['decoder2'],indices['pool1'],tf.shape(layers['conv1']))
layers['decoder1']=deconvLayer(layers['uppool1'],[3,3,3,16],tf.shape(x),name='decoder1',trainable=False,init=inits['decoder1'])


loss=tf.reduce_sum(tf.square(layers['decoder1']-y),axis=3)
loss=tf.reduce_mean(loss)

lr=1e-3
trainStep=tf.train.RMSPropOptimizer(lr,decay=0.9).minimize(loss)
########[small utils]########
for var in tf.global_variables():
	print var.name
	print var.get_shape()


def showFeature(layer,imgshape=[8,8],name='feature-map'):
	com=sess.run(layer,feed_dict={x:[raws[0]],y:[labels[0]]})
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
epoch=35000
test_iter=100
save_iter=100
batchSize=4





with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	print('init done')
	lossAvg=0
	lossNum=0
	for i in range(startPoint-1,epoch):
		names=random.sample(nameList,batchSize)
		raws=[]
		labels=[]
		for name in names:
			raw=np.load(traindir+name)
			raws.append(raw)
			label=np.load(traindir+name)
			labels.append(label)
		sess.run(trainStep,feed_dict={x:raws,y:labels})
		lossResult=sess.run(loss,feed_dict={x:raws,y:labels})
		lossAvg+=lossResult
		lossNum+=1
		print('iter_'+str(i+1)+': '+str(lossResult))
		if (i+1)%save_iter==0:
			print('average loss: %f'%(lossAvg/lossNum))
			param=sess.run(paramDict)
			print np.shape(param['conv1'][0])
			print np.shape(param['conv1'][1])
			print np.shape(param['conv1'][2])

			np.save('./models_4layer/_iter_'+str(i+1)+'.npy',param)
			print('Model saved.')
		if (i+1)%test_iter==0:
#			for var in tf.trainable_variables():
#				print var.name
#				print var.get_shape()
			print('average loss: %f'%(lossAvg/lossNum))
			pred=sess.run(layers['decoder1'],feed_dict={x:[raws[0]],y:[labels[0]]})
			pred_=truncate(pred[0],(0,1))
			predshow=np.uint8(pred_*255)
			labelsshow=np.uint8(labels[0]*255)
			tools.showImage(predshow,'win1')
			tools.showImage(labelsshow,'win2')
			showFeature(layers['pool1'],[4,4],'pool1')
			showFeature(layers['pool2'],[4,4],'pool2')
			showFeature(layers['pool3'],[4,4],'pool3')
			showFeature(layers['pool4'],[4,4],'pool4')
			cv2.waitKey()
			cv2.destroyAllWindows()
			lossAvg=0
			lossNum=0


print('OK')

