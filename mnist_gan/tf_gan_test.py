import tensorflow as tf
import numpy as np
import os
from scipy import misc

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

activation = tf.nn.leaky_relu


def discriminator(data_in, reuse = False):
	with tf.variable_scope('discriminator', reuse=reuse):
		conv1_1 = tf.layers.conv2d(data_in, name = "conv1_1", filters = 16, kernel_size = 5, padding = "same", activation = activation)
		conv1_1 = tf.layers.batch_normalization(conv1_1)
		conv1_2 = tf.layers.conv2d(conv1_1, name = "conv1_2", filters = 32, kernel_size = 5, padding = "same", activation = activation)
		conv1_2 = tf.layers.batch_normalization(conv1_2)
		conv1_3 = tf.layers.conv2d(conv1_2, name = "conv1_3", strides = (2,2), filters = 32, kernel_size = 5, padding = "same", activation = activation)
		conv1_3 = tf.layers.batch_normalization(conv1_3)

		conv2_1 = tf.layers.conv2d(conv1_3, name = "conv2_1", filters = 64, kernel_size = 5, padding = "same", activation = activation)
		conv2_1 = tf.layers.batch_normalization(conv2_1)
		conv2_2 = tf.layers.conv2d(conv2_1, name = "conv2_2", filters = 64, kernel_size = 5, padding = "same", activation = activation)
		conv2_2 = tf.layers.batch_normalization(conv2_2)
		conv2_3 = tf.layers.conv2d(conv2_2, name = "conv2_3", strides = (2,2), filters = 64, kernel_size = 5, padding = "same", activation = activation)
		conv2_3 = tf.layers.batch_normalization(conv2_3)

		flatten = tf.reshape(conv2_3, (-1, 49*64))
		fc1 = tf.layers.dense(flatten, 100, name = "fc1", activation = activation)
		fc1 = tf.layers.batch_normalization(fc1)
		fc2 = tf.layers.dense(fc1, 50, name = "fc2", activation = activation)
		fc2 = tf.layers.batch_normalization(fc2)

		output = tf.layers.dense(fc2, 1, name = "output")
		#output = tf.layers.batch_normalization(output)
		output = tf.nn.sigmoid(output)

		return output

def generator(data_in, reuse = False):
	with tf.variable_scope('generator', reuse=reuse):
		fc1 = tf.layers.dense(data_in, 1000, name = "fc1", activation = activation)
		fc1 = tf.layers.batch_normalization(fc1)
		fc2 = tf.layers.dense(fc1, 7*7*64, name = "fc2", activation = activation)
		fc2 = tf.layers.batch_normalization(fc2)
		
		unflatten = tf.reshape(fc2, (-1, 7, 7, 64))

		upsample1 = tf.image.resize_images(unflatten, (14, 14), method = tf.image.ResizeMethod.BILINEAR)
		conv1_1 = tf.layers.conv2d(upsample1, name = "conv1_1", filters = 64, kernel_size = 5, padding = "same", activation = activation)
		conv1_1 = tf.layers.batch_normalization(conv1_1)
		conv1_2 = tf.layers.conv2d(conv1_1, name = "conv1_2", filters = 64, kernel_size = 5, padding = "same", activation = activation)
		conv1_2 = tf.layers.batch_normalization(conv1_2)

		upsample2 = tf.image.resize_images(conv1_2, (28, 28), method = tf.image.ResizeMethod.BILINEAR)
		conv2_1 = tf.layers.conv2d(upsample2, name = "conv2_1", filters = 64, kernel_size = 5, padding = "same", activation = activation)
		conv2_1 = tf.layers.batch_normalization(conv2_1)
		conv2_2 = tf.layers.conv2d(conv2_1, name = "conv2_2", filters = 64, kernel_size = 5, padding = "same", activation = activation)
		conv2_2 = tf.layers.batch_normalization(conv2_2)

		conv3 = tf.layers.conv2d(conv2_2, name = "conv3", filters = 1, kernel_size = 5, padding = "same")
		conv3 = tf.layers.batch_normalization(conv3)

		output = tf.nn.relu6(conv3) / 6.0

		return output


img_in = tf.placeholder("float", shape = (None, 28, 28, 1))
z_in = tf.placeholder("float", shape = (None, 100))
label = tf.placeholder("float", shape = (None, 10))

G_z = generator(z_in)
D_real_logits = discriminator(img_in / 255.0)
D_fake_logits = discriminator(G_z, reuse=True)


batch_size = 50

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1])))

T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(0.0002).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(0.0002).minimize(G_loss, var_list=G_vars)

restore_path = 'saver_test/'
saver = tf.train.Saver(max_to_keep = 1)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	checkpoint = tf.train.latest_checkpoint(restore_path)
	if checkpoint:
		print("restore from: " + checkpoint)
		saver.restore(sess, checkpoint)

	
	for i in range(10000):
		train_data, label_data = mnist.train.next_batch(batch_size)
		train_data = np.reshape(train_data, (batch_size, 28,28,1))
		
		feed_dict = {img_in: train_data, z_in: np.random.normal(0, 1, (batch_size, 100))}
		_, loss_d_ = sess.run([D_optim, D_loss], feed_dict = feed_dict)

		feed_dict = {img_in: train_data, z_in: np.random.normal(0, 1, (batch_size, 100))}
		loss_g_, _, out_img = sess.run([G_loss, G_optim, G_z], feed_dict = feed_dict)

		if (i%10==0):
			print(loss_d_)
			print(loss_g_)
			print("==============")
			misc.imsave("ans.jpg", np.reshape(out_img[0], (28,28)))
			saver.save(sess, restore_path, global_step = i)