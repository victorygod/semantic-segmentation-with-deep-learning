import tensorflow as tf
import numpy as np
import os, time
from scipy import misc

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

activation = tf.nn.leaky_relu


def discriminator(data_in, reuse = False, batch_norm = True):
	with tf.variable_scope('discriminator', reuse=reuse):
		conv1 = tf.layers.conv2d(data_in, name = "conv1", filters = 128, kernel_size = 5, strides = (2,2), padding = "same", activation = activation)
		conv1 = tf.layers.batch_normalization(conv1, training = True) if batch_norm else conv1
		conv2 = tf.layers.conv2d(conv1, name = "conv2", filters = 256, kernel_size = 5, strides = (2,2), padding = "same", activation = activation)
		conv2 = tf.layers.batch_normalization(conv2, training = True) if batch_norm else conv2
		conv3 = tf.layers.conv2d(conv2, name = "conv3", filters = 512, kernel_size = 5, strides = (2,2), padding = "same", activation = activation)
		conv3 = tf.layers.batch_normalization(conv3, training = True) if batch_norm else conv3
		conv4 = tf.layers.conv2d(conv3, name = "conv4", filters = 1024, kernel_size = 5, strides = (2,2), padding = "same", activation = activation)
		conv4 = tf.layers.batch_normalization(conv4, training = True) if batch_norm else conv4

		conv5 = tf.layers.conv2d(conv4, 1, [2, 2],name = "output", strides=(1, 1), padding='valid')
		output = tf.reshape(conv5, (-1, 1))

		# output = tf.layers.batch_normalization(output, training = True)
		# output = tf.nn.sigmoid(output)

		return output

def generator(data_in, reuse = False, batch_norm = True):
	with tf.variable_scope('generator', reuse=reuse):
		fc1 = tf.layers.dense(data_in, 1024, name = "fc1", activation = activation)
		fc1 = tf.layers.batch_normalization(fc1, training = True) if batch_norm else fc1
		fc2 = tf.layers.dense(fc1, 2*2*64, name = "fc2", activation = activation)
		fc2 = tf.layers.batch_normalization(fc2, training = True) if batch_norm else fc2
		
		unflatten = tf.reshape(fc2, (-1, 2, 2, 64))

		upsample1 = tf.image.resize_images(unflatten, (4, 4), method = tf.image.ResizeMethod.BILINEAR)
		conv1_1 = tf.layers.conv2d(upsample1, name = "conv1_1", filters = 128, kernel_size = 5, padding = "same", activation = activation)
		conv1_1 = tf.layers.batch_normalization(conv1_1, training = True) if batch_norm else conv1_1
		conv1_2 = tf.layers.conv2d(conv1_1, name = "conv1_2", filters = 128, kernel_size = 5, padding = "same", activation = activation)
		conv1_2 = tf.layers.batch_normalization(conv1_2, training = True) if batch_norm else conv1_2

		upsample2 = tf.image.resize_images(conv1_2, (7, 7), method = tf.image.ResizeMethod.BILINEAR)
		conv2_1 = tf.layers.conv2d(upsample2, name = "conv2_1", filters = 128, kernel_size = 5, padding = "same", activation = activation)
		conv2_1 = tf.layers.batch_normalization(conv2_1, training = True) if batch_norm else conv2_1
		conv2_2 = tf.layers.conv2d(conv2_1, name = "conv2_2", filters = 128, kernel_size = 5, padding = "same", activation = activation)
		conv2_2 = tf.layers.batch_normalization(conv2_2, training = True) if batch_norm else conv2_2

		upsample3 = tf.image.resize_images(conv2_2, (14, 14), method = tf.image.ResizeMethod.BILINEAR)
		conv3_1 = tf.layers.conv2d(upsample3, name = "conv3_1", filters = 128, kernel_size = 5, padding = "same", activation = activation)
		conv3_1 = tf.layers.batch_normalization(conv3_1, training = True) if batch_norm else conv3_1
		conv3_2 = tf.layers.conv2d(conv3_1, name = "conv3_2", filters = 128, kernel_size = 5, padding = "same", activation = activation)
		conv3_2 = tf.layers.batch_normalization(conv3_2, training = True) if batch_norm else conv3_2


		upsample4 = tf.image.resize_images(conv3_2, (28, 28), method = tf.image.ResizeMethod.BILINEAR)
		conv4_1 = tf.layers.conv2d(upsample4, name = "conv4_1", filters = 128, kernel_size = 7, padding = "same", activation = activation)
		conv4_1 = tf.layers.batch_normalization(conv4_1, training = True) if batch_norm else conv4_1
		conv4_2 = tf.layers.conv2d(conv4_1, name = "conv4_2", filters = 128, kernel_size = 7, padding = "same", activation = activation)
		conv4_2 = tf.layers.batch_normalization(conv4_2, training = True) if batch_norm else conv4_2

		conv5 = tf.layers.conv2d(conv4_2, name = "conv5", filters = 1, kernel_size = 7, padding = "same")
		# conv5 = tf.layers.batch_normalization(conv5, training = True) if batch_norm else conv5

		output = tf.nn.tanh(conv5)

		# conv1 = tf.layers.conv2d_transpose(tf.reshape(data_in, (-1, 1, 1, 100)), 1024, [4, 4], strides=(1, 1), padding='valid')
		# lrelu1 = activation(tf.layers.batch_normalization(conv1, training=True), 0.2)

		# # 2nd hidden layer
		# conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
		# lrelu2 = activation(tf.layers.batch_normalization(conv2, training=True), 0.2)

		# # 3rd hidden layer
		# conv3 = tf.layers.conv2d(lrelu2, 256, kernel_size = (2,2), strides = (1,1), padding = "valid", activation = activation)
		# lrelu3 = tf.layers.batch_normalization(conv3, training = True)

		# # 4th hidden layer
		# conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
		# lrelu4 = activation(tf.layers.batch_normalization(conv4, training=True), 0.2)

		# # output layer
		# conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
		# output = tf.nn.tanh(conv5)

		return output


img_in = tf.placeholder("float", shape = (None, 28, 28, 1))
z_in = tf.placeholder("float", shape = (None, 100))
#label = tf.placeholder("float", shape = (None, 10))

G_z = generator(z_in)
D_real_logits = discriminator((img_in - 0.5 ) / 0.5, batch_norm = False)
D_fake_logits = discriminator(G_z, reuse=True, batch_norm = False)


batch_size = 50

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1])))

T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(0.0002, beta1 = 0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(0.0002, beta1 = 0.5).minimize(G_loss, var_list=G_vars)

s1 = tf.summary.scalar("loss_G", G_loss)
s2 = tf.summary.scalar("loss_D", D_loss)
s3 = tf.summary.image("ans", G_z)
merged_summary = tf.summary.merge_all()

restore_path = 'saver_test/'
saver = tf.train.Saver(max_to_keep = 1)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#np.random.seed(int(time.time()))
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	checkpoint = tf.train.latest_checkpoint(restore_path)
	if checkpoint:
		print("restore from: " + checkpoint)
		saver.restore(sess, checkpoint)
	
	writer = tf.summary.FileWriter("gan_test/", sess.graph)
	
	for i in range(100000):
		train_data, label_data = mnist.train.next_batch(batch_size)
		train_data = np.reshape(train_data, (batch_size, 28,28,1))
		
		feed_dict = {img_in: train_data, z_in: np.random.normal(0, 1, (batch_size, 100))}
		_, loss_d_ = sess.run([D_optim, D_loss], feed_dict = feed_dict)

		feed_dict = {img_in: train_data, z_in: np.random.normal(0, 1, (batch_size, 100))}
		loss_g_, _, out_img = sess.run([G_loss, G_optim, G_z], feed_dict = feed_dict)

		if (i%5==0):
			print(loss_d_)
			print(loss_g_)
			print("==============")
			misc.imsave("ans.jpg", np.reshape(out_img[0], (28,28)))

			s = sess.run(merged_summary, feed_dict = {img_in: train_data, z_in: np.random.normal(0, 1, (batch_size, 100))})
			writer.add_summary(s, i)
			saver.save(sess, restore_path, global_step = i)