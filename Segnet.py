#-*-coding: utf-8-*-#
import tensorflow as tf
import numpy as np
import os
import random
import tools
import cv2

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
		if init == None:
			kernel=tf.Variable(tf.truncated_normal(shape),trainable=True,name='kernel')
			bias=tf.Variable(tf.zeros(shape[-1]),trainable=True,name='bias')
		else:
			kernel=tf.Variable(tf.constant(init[0]),trainable=True,name='kernel')
			bias=tf.Variable(tf.constant(init[1]),trainable=True,name='bias')			
		fdict=[kernel,bias]
		#variableList[kernel.name]=kernel
		#variableList[bias.name]=bias
		return fdict

def convLayer(inTensor,inFilterShape,activationFunc=tf.nn.relu,name="conv0",trainable=True,init=None):
	#input are a tensor of shape [batch, h, w, channels] and a filter shape of [h, w, ch_in, ch_out]
	with tf.name_scope(name) as scope:
		filterDict=getFilterGroup(inFilterShape,"filter0",trainable=trainable,init=init)
		ans=tf.nn.conv2d(inTensor,filterDict[0],strides=[1,1,1,1],padding="SAME",name="conv2d")
		ans=activationFunc(ans+filterDict[1])
		ans=tf.contrib.layers.batch_norm(ans)

		paramDict[name]=filterDict
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


########[building and initializing]########

tf.reset_default_graph()

#we may change to bunch dim later#
#bunch=20
inc=3
outc=32
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
layers={}
indices={}

restoreMark=True
startPoint=6000
restorePath='/home/mummy/research/Segnet/TF_version/models/20171103/_iter_'+str(startPoint)+'.npy'


# nameMap insideParam:outsideParam
# As you change the model, remember to change the nameMap accordingly
nameMap={'conv1':'conv1'
,'conv2':'conv2'
,'conv3':'conv3'
,'decoder3':'decoder3'
,'decoder2':'decoder2'
,'decoder1':'decoder1'
,'score':'score'}

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



layers['conv1']=convLayer(x,[7,7,3,64],name='conv1',trainable=True,init=inits['conv1'])
layers['pool1'],indices['pool1']=tf.nn.max_pool_with_argmax(layers['conv1'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1')


layers['conv2']=convLayer(layers['pool1'],[7,7,64,64],name='conv2',trainable=True,init=inits['conv2'])
[layers['pool2'],indices['pool2']]=tf.nn.max_pool_with_argmax(layers['conv2'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2')


layers['conv3']=convLayer(layers['pool2'],[7,7,64,64],name='conv3',trainable=True,init=inits['conv3'])
[layers['pool3'],indices['pool3']]=tf.nn.max_pool_with_argmax(layers['conv3'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool3')
'''
layers['conv4']=convLayer(layers['pool3'],[7,7,64,64],name='conv4',trainable=False,init['conv4'])
[layers['pool4'],indices['pool4']]=tf.nn.max_pool_with_argmax(layers['conv4'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool4')

layers['uppool4']=upsamplingLayer(layers['pool4'],indices['pool4'],tf.shape(layers['conv4']))
layers['decoder4']=convLayer(layers['uppool4'],[7,7,64,64],name='decoder4',trainable=False,inits['decoder4'])
'''
layers['uppool3']=upsamplingLayer(layers['pool3'],indices['pool3'],tf.shape(layers['conv3']))
layers['decoder3']=convLayer(layers['uppool3'],[7,7,64,64],name='decoder3',trainable=True,init=inits['decoder3'])

layers['uppool2']=upsamplingLayer(layers['decoder3'],indices['pool2'],tf.shape(layers['conv2']))
layers['decoder2']=convLayer(layers['uppool2'],[7,7,64,64],name='decoder2',trainable=True,init=inits['decoder2'])

layers['uppool1']=upsamplingLayer(layers['decoder2'],indices['pool1'],tf.shape(layers['conv1']))
layers['decoder1']=convLayer(layers['uppool1'],[7,7,64,64],name='decoder1',trainable=True,init=inits['decoder1'])

layers['score']=convLayer(layers['decoder1'],[1,1,64,32],name='score',trainable=True,init=inits['score'])
layers['softmax']=tf.nn.softmax(layers['score'])

loss=-tf.reduce_sum(tf.mul(layers['softmax'],tf.log(y+1e-10)),axis=3)
loss=tf.reduce_mean(loss)

lr=1e-3
trainStep=tf.train.RMSPropOptimizer(lr,decay=0.9).minimize(loss)

########[training]########

traindir='/home/mummy/research/Segnet/dataset/CamVid/samples/'
testdir=''
nameList=os.listdir(traindir+'raw')
epoch=30000
test_iter=10
save_iter=100
batchSize=5



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
			raw=np.load(traindir+'raw/'+name)
			rawsh=np.shape(raw)
			raw_=np.zeros(rawsh,np.float32)
			raw_=raw
			raws.append(raw_)
			label=np.load(traindir+'label/'+name)
			labelsh=np.shape(label)
			label_=np.zeros(labelsh,np.float32)
			label_=label
			labels.append(label_)
		sess.run(trainStep,feed_dict={x:raws,y:labels})
		lossResult=sess.run(loss,feed_dict={x:raws,y:labels})
		lossAvg+=lossResult
		lossNum+=1
		print('iter_'+str(i+1)+': '+str(lossResult))
		if (i+1)%save_iter==0:
			param=sess.run(paramDict)
			np.save('/home/mummy/research/Segnet/TF_version/models/20171103/_iter_'+str(i+1)+'.npy',param)
			print('Model saved.')
		if (i+1)%test_iter==0:
			lossAvg=lossAvg/lossNum
			print('average loss: %f'%lossAvg)
			pred=sess.run(layers['softmax'],feed_dict={x:[raws[0]],y:[labels[0]]})
			ans1=tools.label2img(pred[0])
			ans2=tools.label2img(labels[0])
			np.shape(ans1)
			np.shape(ans2)
			tools.showImage(ans1,'win1')
			tools.showImage(ans2,'win2')
			cv2.waitKey()
			cv2.destroyAllWindows()
			lossAvg=0
			lossNum=0



print('OK')
