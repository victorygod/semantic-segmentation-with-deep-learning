import tensorflow as tf
import numpy as np
import os
import math
from scipy import misc
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

BATCH_SIZE = 20
mem_size = 3

def get_kernel(name, shape = (7,7,64,64), trainable = True):
	init = tf.constant_initializer(value=data[name+'/kernel:0'], dtype=tf.float32) if name+'/kernel:0' in data else tf.truncated_normal_initializer(stddev=0.1)
	return tf.get_variable(name = "kernel", initializer = init, shape = shape, trainable = trainable)

def get_bias(name, shape = [64], trainable = True):
	init = tf.constant_initializer(value=data[name+'/bias:0'], dtype=tf.float32) if name+'/bias:0' in data else tf.constant_initializer(0.0)
	return tf.get_variable(name = "bias", initializer = init, shape = shape, trainable = trainable)

def conv_layer(in_data, name, kernel_shape = (7,7,64,64), trainable = True, has_bias = True):
	with tf.variable_scope(name) as scope:
		kernel = get_kernel(name, shape = kernel_shape, trainable = trainable)
		bias = get_bias(name,shape = [kernel_shape[3]], trainable = trainable)
		conv = tf.nn.conv2d(in_data, kernel, strides = [1,1,1,1], padding = 'SAME')
		if has_bias:
		    conv = tf.nn.bias_add(conv, bias)
		conv = batch_norm_layer(conv, name)
		return conv

def mem_layer(name ,shape, trainable = True):
	with tf.variable_scope(name) as scope:
		in_data = tf.zeros([shape[0], shape[1]*shape[2]*shape[3]])
		bias = get_bias(name, shape = [shape[1]*shape[2]*shape[3]], trainable = trainable)
		layer = tf.nn.bias_add(in_data, bias)
		layer = tf.reshape(layer, shape)
		return layer

def batch_norm_layer(inputT, name):
	gamma = tf.constant_initializer(value=data[name+'/batch_normalization/gamma:0'], dtype=tf.float32) if name+'/batch_normalization/gamma:0' in data else tf.ones_initializer()
	beta = tf.constant_initializer(value=data[name + '/batch_normalization/beta:0'], dtype=tf.float32) if name + '/batch_normalization/beta:0' in data else tf.zeros_initializer()
	moving_mean = tf.constant_initializer(value=data[name + '/batch_normalization/moving_mean:0'], dtype=tf.float32) if name + '/batch_normalization/moving_mean:0' in data else tf.zeros_initializer()
	moving_variance = tf.constant_initializer(value=data[name+'/batch_normalization/moving_variance:0'], dtype=tf.float32) if name+'/batch_normalization/moving_variance:0' in data else tf.ones_initializer()
	
	return tf.layers.batch_normalization(inputT, training = True, gamma_initializer = gamma, beta_initializer = beta, moving_variance_initializer = moving_variance, moving_mean_initializer = moving_mean)
	
	#return tf.contrib.layers.batch_norm(inputT, is_training=True, center=False, updates_collections=None, scope=scope+"_bn")

def fc_layer(in_data, name, shape, trainable = True):
	with tf.variable_scope(name) as scope:
		kernel = get_kernel(name, shape = shape, trainable = trainable)
		bias = get_bias(name, shape = [shape[1]], trainable = trainable)
		fc = tf.matmul(in_data, kernel) + bias
		fc = batch_norm_layer(fc, name)
		return fc

def max_pool_2x2(x, name, size = 2):
    return tf.nn.max_pool(x, ksize = [1,size,size,1], strides = [1,size,size,1], padding = 'SAME')

def get_deconv_kernel(name, shape, trainable = True):
	if name+'/deconv_kernel:0' in data:
		init = tf.constant_initializer(value=data[name+'/deconv_kernel:0'], dtype=tf.float32)
	else:
		kernel = np.zeros(shape)
		width = shape[0]
		height = shape[1]
		f = math.ceil(width/2.0)
		c = (2 * f - 1 - f % 2) / (2.0 * f)
		bilinear = np.zeros((shape[0], shape[1]))
		for x in range(width):
		    for y in range(height):
		        bilinear[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
		for i in range(shape[2]):
		    kernel[:,:,i,i] = bilinear
		init = tf.constant_initializer(value=kernel, dtype=tf.float32)

	return tf.get_variable(name = "deconv_kernel", initializer = init, shape = shape, trainable = trainable)

def get_deconv_layer(in_data, name, shape, kernel_size = 4, stride = 2, trainable = True):
	with tf.variable_scope(name) as scope:
		strides = [1, stride, stride, 1]
		kernel_shape = [kernel_size, kernel_size, shape[3], in_data.shape[3]]
		kernel = get_deconv_kernel(name, kernel_shape, trainable = trainable)
		deconv = tf.nn.conv2d_transpose(in_data, kernel, shape, strides = strides, padding = "SAME")
		return deconv

def max_pool_2x2_with_argmax(x, name):
	return tf.nn.max_pool_with_argmax(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = name)

def uppooling_layer(indata, raveled_argmax, output_shape):
	input_shape =  tf.shape(indata)
	flat_input_size = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3]
	flat_output_shape = (output_shape[0], output_shape[1] * output_shape[2] * output_shape[3])

	values = tf.reshape(indata, [flat_input_size])
	batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=raveled_argmax.dtype), shape=(input_shape[0], 1, 1, 1))
	batch_num = tf.ones_like(raveled_argmax) * batch_range
	batch_num = tf.reshape(batch_num, (flat_input_size, 1))
	indices = tf.reshape(raveled_argmax, (flat_input_size, 1))
	indices = tf.concat([batch_num, indices], 1)

	output = tf.scatter_nd(indices, values, shape=tf.cast(flat_output_shape, tf.int64))
	output = tf.reshape(output, output_shape)
	return output


def activateFunc(x):
	return tf.nn.relu(x)

def linear_truncate(x, scope):
	return tf.maximum(tf.minimum(x, scope[1]), scope[0])

recovery_path = './autoencoder_mem.npy'
data = {}
start_epoch = 0
if os.path.isfile(recovery_path):
    data = np.load(recovery_path).item()
    if "epoch" in data:
        start_epoch = data["epoch"] + 1
    print("Recover from " + recovery_path + ". at epoch " + str(start_epoch))

AUTOENCODER_TRAINABLE = True


img_in = tf.placeholder("float")

labels = tf.placeholder("float")

conv1_1 = conv_layer(img_in, "conv1_1", kernel_shape = (7,7,3,10), trainable = AUTOENCODER_TRAINABLE)
conv1_1 = activateFunc(conv1_1) 
conv1_2 = conv_layer(conv1_1, "conv1_2", kernel_shape = (7,7,10,10), trainable = AUTOENCODER_TRAINABLE)
conv1_2 = activateFunc(conv1_2)
pool1= max_pool_2x2(conv1_2, "pool1", size = 4)
conv2_1 = conv_layer(pool1, "conv2_1", kernel_shape = (7,7,10,10), trainable = AUTOENCODER_TRAINABLE)
conv2_1 = activateFunc(conv2_1)
conv2_2 = conv_layer(conv2_1, "conv2_2", kernel_shape = (7,7,10,10), trainable = AUTOENCODER_TRAINABLE)
conv2_2 = activateFunc(conv2_2)
pool2 = max_pool_2x2(conv2_2, "pool2", size = 4)


flat_layer = tf.reshape(pool2, (BATCH_SIZE, 1800))
fc1 = fc_layer(flat_layer, "fc1", (1800, 5), trainable = AUTOENCODER_TRAINABLE)
fc1 = activateFunc(fc1)
fc2 = fc_layer(fc1, "fc2", (5,1800), trainable = AUTOENCODER_TRAINABLE)
fc2 = activateFunc(fc2)
unflat_layer = tf.reshape(fc2, tf.shape(pool2))


uppooling_4 = get_deconv_layer(unflat_layer, name = "deconv4", shape = (BATCH_SIZE, 23, 30, 10), trainable = AUTOENCODER_TRAINABLE)
decoder_conv4_1 = conv_layer(uppooling_4, "decoder_conv4_1", kernel_shape = (7,7,10,64), trainable = AUTOENCODER_TRAINABLE)
# decoder_conv4_2 = conv_layer(decoder_conv4_1, "decoder_conv4_2", trainable = AUTOENCODER_TRAINABLE)
# decoder_conv4_2 = activateFunc(decoder_conv4_2)

mem_layer3 = mem_layer("mem_layer3", shape = (BATCH_SIZE,  23, 30, mem_size), trainable = AUTOENCODER_TRAINABLE)

fuse_layer3 = tf.concat([decoder_conv4_1, mem_layer3], axis = 3)
fuse_layer3 = conv_layer(fuse_layer3, "fuse_layer3", kernel_shape = (7,7,64+mem_size,64), trainable = AUTOENCODER_TRAINABLE)
fuse_layer3 = activateFunc(fuse_layer3)
#fuse_layer3 = decoder_conv4_2 + mem_layer3

uppooling_3 = get_deconv_layer(fuse_layer3, name = "deconv3", shape = (BATCH_SIZE,  45, 60, 64), trainable = AUTOENCODER_TRAINABLE) 
# decoder_conv3_1 = conv_layer(uppooling_3, "decoder_conv3_1", trainable = AUTOENCODER_TRAINABLE)
# decoder_conv3_2 = conv_layer(decoder_conv3_1, "decoder_conv3_2", trainable = AUTOENCODER_TRAINABLE)
# decoder_conv3_2 = activateFunc(decoder_conv3_2)

mem_layer2 = mem_layer("mem_layer2", shape = (BATCH_SIZE,  45, 60, mem_size), trainable = AUTOENCODER_TRAINABLE)

fuse_layer2 = tf.concat([uppooling_3, mem_layer2], axis = 3)
fuse_layer2 = conv_layer(fuse_layer2, "fuse_layer2", kernel_shape = (7,7,64+mem_size,64), trainable = AUTOENCODER_TRAINABLE)
fuse_layer2 = activateFunc(fuse_layer2)
#fuse_layer2 = decoder_conv3_2 + mem_layer2

uppooling_2 = get_deconv_layer(fuse_layer2, name = "deconv2", shape = (BATCH_SIZE, 90, 120, 64), trainable = True)
# decoder_conv2_1 = conv_layer(uppooling_2, "decoder_conv2_1", trainable = AUTOENCODER_TRAINABLE)
# decoder_conv2_2 = conv_layer(decoder_conv2_1, "decoder_conv2_2", trainable = AUTOENCODER_TRAINABLE)
# decoder_conv2_2 = activateFunc(decoder_conv2_2)

mem_layer1 = mem_layer("mem_layer1", shape = (BATCH_SIZE, 90, 120, mem_size), trainable = AUTOENCODER_TRAINABLE)

fuse_layer1 = tf.concat([uppooling_2, mem_layer1], axis = 3)
fuse_layer1 = conv_layer(fuse_layer1, "fuse_layer1", kernel_shape = (7,7,64+mem_size,64), trainable = AUTOENCODER_TRAINABLE)
fuse_layer1 = activateFunc(fuse_layer1)
#fuse_layer1 = decoder_conv2_2 + mem_layer2

uppooling_1 = get_deconv_layer(fuse_layer1, name = "deconv1", shape = (BATCH_SIZE, 180, 240, 64), trainable = True)
# decoder_conv1_1 = conv_layer(uppooling_1, "decoder_conv1_1", trainable = AUTOENCODER_TRAINABLE)
# decoder_conv1_2 = conv_layer(decoder_conv1_1, "decoder_conv1_2", trainable = AUTOENCODER_TRAINABLE)
# decoder_conv1_2 = activateFunc(decoder_conv1_2)

mem_layer0 = mem_layer("mem_layer0", shape = (BATCH_SIZE, 180, 240, mem_size), trainable = AUTOENCODER_TRAINABLE)

fuse_layer0 = tf.concat([uppooling_1, mem_layer0], axis = 3)
fuse_layer0 = conv_layer(fuse_layer0, "fuse_layer0", kernel_shape = (7,7,64+mem_size,64), trainable = AUTOENCODER_TRAINABLE)
fuse_layer0 = activateFunc(fuse_layer0)

output = conv_layer(fuse_layer0, "score", kernel_shape = (7,7,64,3), trainable = AUTOENCODER_TRAINABLE)
output = linear_truncate(output, (0,1)) * 255 #tf.nn.sigmoid(output) * 255 #linear_truncate(output, (0,255))

output2 = tf.abs(img_in-output)
#output2 = tf.abs(output2 - tf.reduce_mean(output2))

loss = tf.reduce_sum(tf.square(labels-output), axis = 3)
loss = tf.reduce_mean(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
	train_step = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)#tf.train.RMSPropOptimizer(learning_rate = 0.01, decay = 0.99).minimize(loss)


with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	train_dir = '../GWU/2015.02/'
	data_list = os.listdir(train_dir)
	l = len(data_list)-1

	output_tensors = [train_step, loss, output, output2]

	output_parameters = []
	for variables in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
		if variables.name.find("Adam")==-1:
			output_parameters.append(variables.name)
	print(output_parameters)

	for epoch in range(start_epoch, 180):
		temp_loss = 0
		num = 0
		h = np.arange(l)
		i = 0
		while l-i>BATCH_SIZE:
			batch_data = []
			batch_label = []
			for j in range(BATCH_SIZE):
				m = np.random.randint(l-i, size = 1)[0]
				index = h[m]
				h[m] = h[l-i-1]
				img = misc.imread(os.path.join(train_dir+ data_list[index]))
				img = misc.imresize(img, (180,240))
				label = img
				batch_data.append(img)
				batch_label.append(label)
				i+=1

			feed_dict = {img_in: batch_data, labels: batch_label}
			output_values = sess.run(output_tensors, feed_dict = feed_dict)

			print(str(epoch) + " " + str(i) + ' ' + str(output_values[1]))
			print(output_values[2][0,150,150,:])
			print(batch_label[0][150,150,:])

			print("========================================")

			num+=1
			temp_loss = ((num-1)/num) * temp_loss + output_values[1]/num

			if i % (BATCH_SIZE*2) == 0: 
				misc.imsave("ans.jpg", output_values[2][0])
				misc.imsave("difference.jpg", output_values[3][0])
				misc.imsave("img.jpg", batch_data[0])

		data={}
		output_values = sess.run(output_parameters)
		for i in range(len(output_parameters)):
			data[output_parameters[i]] = output_values[i]

		print("avg_loss = " + str(temp_loss))
		data["avg_loss"] = temp_loss
		data["epoch"] = epoch
		np.save(recovery_path, data)

		test_file1 = "../GWU/2015.01/20150103_120040.jpg"
		test_file2 = "../GWU/2015.01/20150116_220121.jpg"
		test_file3 = "../GWU/2015.01/20150117_153055.jpg"
		test_file4 = "../GWU/2015.01/20150131_110043.jpg"
		
		batch_data = [misc.imresize(misc.imread(test_file1), (180,240)), misc.imresize(misc.imread(test_file2), (180,240)),misc.imresize(misc.imread(test_file3), (180,240)), misc.imresize(misc.imread(test_file4), (180,240))]
		for i in range(16):
			batch_data.append(misc.imresize(misc.imread(test_file1), (180,240)))
		feed_dict = {img_in: batch_data, labels: batch_data}
		output_values = sess.run([loss, output, output2], feed_dict = feed_dict)
		print("test_loss = ", str(output_values[0]))
		for i in range(4):
			misc.imsave("test_ans"+str(i)+".jpg", output_values[1][i])
			misc.imsave("test_difference"+str(i)+".jpg", output_values[2][i])
			misc.imsave("test_img"+str(i)+".jpg", batch_data[i])