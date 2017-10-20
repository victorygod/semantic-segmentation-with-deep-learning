# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:47:24 2017

@author: god
"""

import tensorflow as tf
import numpy as np
import os
import math
import utils
import matplotlib.image as mpimg
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

Kernel = {}
Bias = {}

NUM_CLASS = utils.NUM_CLASS

#====================================== functions ==================================================================

def get_kernel(name, shape = (7, 7, 64, 64), trainable = True):
    if name in data:
        init = tf.constant_initializer(value=data[name][0], dtype=tf.float32)
        shape = data[name][0].shape
        return tf.get_variable(name = "kernel", initializer = init, shape = shape, trainable = trainable)
    else:
        init = tf.truncated_normal_initializer(stddev=0.1)
        return tf.get_variable(name = "kernel", initializer = init, shape = shape, trainable = trainable)
    

def get_bias(name, shape = [64], trainable = True):
    if name in data:
        init = tf.constant_initializer(value=data[name][1], dtype=tf.float32)
        shape = data[name][1].shape
        return tf.get_variable(name = "bias", initializer = init, shape = shape, trainable = trainable)
    else:
        bias_init = tf.constant(0.0, shape = shape)
        return tf.get_variable(name = "bias", initializer = bias_init, trainable = trainable)


def conv_layer(in_data, name, kernel_shape = (7,7,64,64), trainable = True, has_bias = True):
    with tf.variable_scope(name) as scope:
        Kernel[name] = get_kernel(name, shape = kernel_shape, trainable = trainable)
        Bias[name] = get_bias(name,shape = [kernel_shape[3]], trainable = trainable)
        conv = tf.nn.conv2d(in_data, Kernel[name], strides = [1,1,1,1], padding = 'SAME')
        if has_bias:
            conv = tf.nn.bias_add(conv, Bias[name])
        conv = batch_norm_layer(conv, True, name)
        return conv

def batch_norm_layer(inputT, is_training, scope):
  return tf.contrib.layers.batch_norm(inputT, is_training=True, center=False, updates_collections=None, scope=scope+"_bn")

def max_pool_2x2(x, name):
    return tf.nn.max_pool_with_argmax(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = name)

#Thanks to https://github.com/tensorflow/tensorflow/issues/2169
def uppooling_layer(indata, raveled_argmax, output_shape):
    input_shape =  tf.shape(indata)
    flat_input_size = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3]
    flat_output_shape = (output_shape[0], output_shape[1] * output_shape[2] * output_shape[3])

    values = tf.reshape(indata, [flat_input_size])
    batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=raveled_argmax.dtype), 
                                      shape=(input_shape[0], 1, 1, 1))
    batch_num = tf.ones_like(raveled_argmax) * batch_range
    batch_num = tf.reshape(batch_num, (flat_input_size, 1))
    indices = tf.reshape(raveled_argmax, (flat_input_size, 1))
    indices = tf.concat([batch_num, indices], 1)

    output = tf.scatter_nd(indices, values, shape=tf.cast(flat_output_shape, tf.int64))
    output = tf.reshape(output, output_shape)
    return output

def channel_change(in_data, name, out_channel = NUM_CLASS, trainable = True):
    with tf.variable_scope(name) as scope:
        in_channel = in_data.get_shape()[3].value
        kernel_shape = [1, 1, in_channel, out_channel]

        Kernel[name] = get_kernel(name, kernel_shape, trainable = trainable)
        Bias[name] = get_bias(name, [out_channel], trainable = trainable)

        conv = tf.nn.conv2d(in_data, Kernel[name], strides = [1,1,1,1], padding = 'SAME')
        conv = conv + Bias[name]
        return conv

def activateFunc(x):
    return tf.nn.relu(x)
#=============================== get_data ===============================================


recovery_path = './net_weights.npy'
data = {}
#start_num = 0
start_epoch = 0
if os.path.isfile(recovery_path):
    data = np.load(recovery_path).item()
    # if "num" in data:
    #     start_num = data["num"] + 1
    # else:
    #     start_num = 0
    if "epoch" in data:
        start_epoch = data["epoch"] + 1
    print("Recover from " + recovery_path + ". at epoch " + str(start_epoch))
else:
    file_path = "./vgg16.npy"
    vgg = np.load(file_path, encoding='latin1').item()
    data["conv1_1"] = vgg["conv1_1"]
    data["conv1_2"] = vgg["conv1_2"]
    data["conv2_1"] = vgg["conv2_1"]
    data["conv2_2"] = vgg["conv2_2"]
    data["conv3_1"] = vgg["conv3_1"]
    data["conv3_2"] = vgg["conv3_2"]
    data["conv3_3"] = vgg["conv3_2"]
    data["conv4_1"] = vgg["conv4_1"]
    data["conv4_2"] = vgg["conv4_2"]
    data["conv4_3"] = vgg["conv4_3"]
    data["conv5_1"] = vgg["conv5_1"]
    data["conv5_2"] = vgg["conv5_2"]
    data["conv5_3"] = vgg["conv5_3"]
   
#===================================== build ================================================

in_data = tf.placeholder("float")
labels = tf.placeholder("float")

VGG_MEAN = [103.939, 116.779, 123.68]
red, green, blue = tf.split(in_data, 3, 3)
bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], 3)

encoder_conv1_1 = conv_layer(bgr, "conv1_1", trainable = False)
encoder_conv1_1 = activateFunc(encoder_conv1_1)
encoder_conv1_2 = conv_layer(encoder_conv1_1, "conv1_2", trainable = False)
encoder_conv1_2 = activateFunc(encoder_conv1_2)
pool1, pool1_argmax = max_pool_2x2(encoder_conv1_2, "pool1")

encoder_conv2_1 = conv_layer(pool1, "conv2_1", trainable = False)
encoder_conv2_1 = activateFunc(encoder_conv2_1)
encoder_conv2_2 = conv_layer(encoder_conv2_1, "conv2_2", trainable = False)
encoder_conv2_2 = activateFunc(encoder_conv2_2)
pool2, pool2_argmax = max_pool_2x2(encoder_conv2_2, "pool2")

encoder_conv3_1 = conv_layer(pool2, "conv3_1", trainable = False)
encoder_conv3_1 = activateFunc(encoder_conv3_1)
encoder_conv3_2 = conv_layer(encoder_conv3_1, "conv3_2", trainable = False)
encoder_conv3_2 = activateFunc(encoder_conv3_2)
encoder_conv3_3 = conv_layer(encoder_conv3_2, "conv3_3", trainable = False)
encoder_conv3_3 = activateFunc(encoder_conv3_3)
pool3, pool3_argmax = max_pool_2x2(encoder_conv3_3, "pool3")

encoder_conv4_1 = conv_layer(pool3, "conv4_1", trainable = False)
encoder_conv4_1 = activateFunc(encoder_conv4_1)
encoder_conv4_2 = conv_layer(encoder_conv4_1, "conv4_2", trainable = False)
encoder_conv4_2 = activateFunc(encoder_conv4_2)
encoder_conv4_3 = conv_layer(encoder_conv4_2, "conv4_3", trainable = False)
encoder_conv4_3 = activateFunc(encoder_conv4_3)
pool4, pool4_argmax = max_pool_2x2(encoder_conv4_3, "pool4")

#======================================================

uppooling_4 = uppooling_layer(pool4, pool4_argmax, tf.shape(encoder_conv4_3))
decoder_conv4_1 = conv_layer(uppooling_4, "decoder_conv4_1", kernel_shape = (7,7,512,256), trainable = True)
decoder_conv4_2 = conv_layer(decoder_conv4_1, "decoder_conv4_2", kernel_shape = (7,7,256,256), trainable = True)

uppooling_3 = uppooling_layer(decoder_conv4_2, pool3_argmax, tf.shape(encoder_conv3_3))
decoder_conv3_1 = conv_layer(uppooling_3, "decoder_conv3_1", kernel_shape = (7,7,256,128), trainable = True)
decoder_conv3_2 = conv_layer(decoder_conv3_1, "decoder_conv3_2", kernel_shape = (7,7,128,128), trainable = True)

uppooling_2 = uppooling_layer(decoder_conv3_1, pool2_argmax, tf.shape(encoder_conv2_2))
decoder_conv2_1 = conv_layer(uppooling_2, "decoder_conv2_1", kernel_shape = (7,7,128,128), trainable = True)
decoder_conv2_2 = conv_layer(decoder_conv2_1, "decoder_conv2_2", kernel_shape = (7,7,128,64), trainable = True)

uppooling_1 = uppooling_layer(decoder_conv2_2, pool1_argmax, tf.shape(encoder_conv1_2))
decoder_conv1_1 = conv_layer(uppooling_1, "decoder_conv1_1", kernel_shape = (7,7,64,64), trainable = True)
decoder_conv1_2 = conv_layer(decoder_conv1_1, "decoder_conv1_2", kernel_shape = (7,7,64,64), trainable = True)

output = channel_change(decoder_conv1_2, "score", trainable = False)

output_labels = tf.nn.softmax(output)

class_balancing = np.array([0.421237196, 0.0321783686, 0.00393296868, 0.0458287878, 9.065720940000001e-05,
 0.00060617061, 0.0685252305, 0.0734163681, 0.00209069272, 0.00144482903, 0.00121322838, 0.149995497, 0.00351557839, 
 0.205430693, 0.00517187333, 0.006170700560000001, 0.00322863048, 7.56257441e-05, 0.00808581352, 0.000325119097, 
 0.0174137646, 0.000130203098, 0.00294008424, 0.63458242, 0.0055382111999999995, 1, 0.00019776375099999998, 
 0.00383764581, 1, 0.00270661894, 0.000686221625, 0.00152857153])
median = np.median(class_balancing)
class_balancing/=median
#class_balancing = np.sqrt(class_balancing)

cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(output_labels + 1e-10), class_balancing), axis = 3)
#cross_entropy = -tf.reduce_sum(labels * tf.log(output_labels + 1e-10), axis = 3)
cross_entropy = tf.reduce_mean(cross_entropy)

fine_tune_lr = 0.00001
training_lr = 0.01
train_step = tf.train.RMSPropOptimizer(learning_rate = fine_tune_lr, decay = 0.9).minimize(cross_entropy)

# new_variables = [Kernel["decoder_conv2_1"], Kernel["decoder_conv2_2"], Bias["decoder_conv2_2"], Bias["decoder_conv2_2"]]
# fine_tune_variables = [Kernel["score"], Bias["score"]] #, Kernel["decoder_conv1_1"], Kernel["decoder_conv1_2"], Bias["decoder_conv1_2"], Bias["decoder_conv1_2"]

# global_step = tf.Variable(0, trainable = False)
# learning_rate = tf.train.exponential_decay(0.00001, global_step, 5, 0.96, staircase = True)
# fine_tune_op = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy, var_list = fine_tune_variables, global_step = global_step)
# new_op = tf.train.RMSPropOptimizer(learning_rate = 0.001, decay = 0.9).minimize(cross_entropy, var_list = new_variables)

# train_step = tf.group(fine_tune_op, new_op)

#=================================== run ==================================================

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    label_dir = "./label_small_size/"
    train_dir = './train_small_size/'
    data_list = os.listdir(train_dir)
    label_list = os.listdir(label_dir)

    output_tensors = [train_step, cross_entropy, output_labels]
    for key in Kernel:
        output_tensors.append(Kernel[key])
    for key in Bias:
        output_tensors.append(Bias[key])
    

    l = len(data_list)

    BATCH_SIZE = 20

    for epoch in range(start_epoch, 80):
        i = 0
        h = np.arange(l)
        while i<l:
            start_num = i
            batch_data = []
            batch_label = []
            while i < start_num + BATCH_SIZE and i<l:
                m = np.random.randint(l-i, size = 1)[0]
                index = h[m]
                h[m] = h[l - i - 1]
                img = np.load(train_dir+ data_list[index])
                #labelFileName = data_list[i][0:len(data_list[i])-4]
                label = np.load(label_dir + label_list[index])
                batch_data.append(img)
                batch_label.append(label)
                i+=1
            
            feed_dict = {in_data: batch_data, labels: batch_label}
            output_values = sess.run(output_tensors, feed_dict = feed_dict)

            print(str(epoch) + " " + str(i) + ' ' + str(output_values[1]))
            print("========================================")
            
            j = 3
            for key in Kernel:
                data[key] = [output_values[j]]
                j+=1
            for key in Bias:
                data[key].append(output_values[j])
                j+=1
                
            if (i%40 == 0):
                ans_img = utils.change_label_to_img(output_values[2])
                mpimg.imsave("ans.jpg", ans_img)
                ans_img = utils.change_label_to_img(np.array(batch_label))
                mpimg.imsave("label.jpg", ans_img)

        data["epoch"] = epoch
        np.save(recovery_path, data)