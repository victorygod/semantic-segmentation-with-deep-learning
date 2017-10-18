import tensorflow as tf
import numpy as np
import os
import math
import utils
import matplotlib.image as mpimg
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

BATCH_SIZE = 20
NUM_CLASS = utils.NUM_CLASS
checkpoint_dir = "./checkpoint/"
data = {}

# def get_kernel(name):
# 	if name+"/kernel:0" in data:
# 		print(name+"/kernel:0")
# 		return data[name+"/kernel:0"]
# 	init = tf.truncated_normal_initializer(stddev=0.1)
# 	return tf.get_variable(name = "kernel", initializer = init, shape = [NUM_CLASS])

def single_weight_layer(in_data, name):
	with tf.variable_scope(name) as scope:
		
		init = tf.truncated_normal_initializer(stddev=0.1)
		weight = tf.get_variable(name = "kernel", initializer = init, shape = [NUM_CLASS])
		init = tf.constant(0.0, shape = [NUM_CLASS])
		bias = tf.get_variable(name = "bias", initializer = init)
		layer = tf.reduce_mean(in_data, 0)
		layer = tf.multiply(layer, weight)
		layer = tf.nn.bias_add(layer, bias)
		return layer

# vgg_saver = tf.train.import_meta_graph(checkpoint_dir + 'model.meta')
# vgg_graph = tf.get_default_graph()
# collection = vgg_graph.get_collection("variables")
# for t in collection:
# 	data[t.name] = t

# print(data)

in_data = tf.placeholder("float")
labels = tf.placeholder("float")

layer = single_weight_layer(in_data, "single")

output = tf.nn.softmax(layer)

cross_entropy = -tf.reduce_sum(labels*tf.log(output + 1e-10), [0,3])
cross_entropy = tf.reduce_mean(cross_entropy)
#cross_entropy = cross_entropy / tf.to_float(tf.shape(in_data)[0])

train_step = tf.train.RMSPropOptimizer(learning_rate = 0.01, decay = 0.9).minimize(cross_entropy)

# xxx = tf.get_variable(name="xxxx", initializer = tf.zeros([2]))

# inc_op = tf.assign_add(xxx, [1,1])

with tf.Session() as sess:
	
	init = tf.global_variables_initializer()
	sess.run(init)

	saver = tf.train.Saver()
	step = 0
	ckpt = tf.train.latest_checkpoint(checkpoint_dir)
	if ckpt:
		saver.restore(sess, ckpt)

	label_dir = "./label_small_size/"
	label_list = os.listdir(label_dir)
	l = len(label_list)
	for epoch in range(step, 5):
		i = 0
		
		while i<l:
			start_num = i
			batch_label = []
			while i < start_num + BATCH_SIZE and i<l:
				label = np.load(label_dir + label_list[i])
				batch_label.append(label)
				i+=1

			feed_dict = {in_data: batch_label, labels: batch_label}
			_, c, out = sess.run([train_step, cross_entropy, output], feed_dict = feed_dict)
			print(str(epoch) + ' ' + str(i) + ' ' + str(c))

			if (i%40 == 0):
				ans_img = utils.change_label_to_img(np.array([out]))
				mpimg.imsave("bg.jpg", ans_img)
				ans_img = utils.change_label_to_img(np.array(batch_label))
				mpimg.imsave("label.jpg", ans_img)

		saver.save(sess, checkpoint_dir+"model")
		print(saver.last_checkpoints)


# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# print_tensors_in_checkpoint_file(file_name=checkpoint_dir+'model', tensor_name='', all_tensors=False)