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

NUM_CLASS = 21

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


def conv_layer(in_data, name, kernel_shape = (7,7,64,64), trainable = True):
    with tf.variable_scope(name) as scope:
        Kernel[name] = get_kernel(name, shape = kernel_shape, trainable = trainable)
        Bias[name] = get_bias(name, trainable = trainable)
        conv = tf.nn.conv2d(in_data, Kernel[name], strides = [1,1,1,1], padding = 'SAME')
        conv = tf.nn.bias_add(conv, Bias[name])
        conv = batch_norm_layer(conv, True, name)
        return conv

def batch_norm_layer(inputT, is_training, scope):
  return tf.contrib.layers.batch_norm(inputT, is_training=True, center=False, updates_collections=None, scope=scope+"_bn")

def max_pool_2x2(x, name):
    return tf.nn.max_pool_with_argmax(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME', name = name)

#===============uppooling===================
#CPU only!
#Only for batch_size = 1
def uppooling_layer(indata, raveled_argmax, out_shape):
    shape_argmax = tf.shape(raveled_argmax)
    indices = tf.reshape(raveled_argmax, (shape_argmax[0]*shape_argmax[1]*shape_argmax[2]*shape_argmax[3], 1))
    
    shape = tf.shape(indata)
    values = tf.reshape(indata, (1, shape[0]*shape[1]*shape[2]*shape[3]))
    values = tf.squeeze(values)

    delta = tf.SparseTensor(indices, values, [tf.to_int64((out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]))])
    return tf.reshape(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), out_shape)
#==========================================================================

def channel_change(in_data, name, out_channel = NUM_CLASS, trainable = True):
    with tf.variable_scope(name) as scope:
        in_channel = in_data.get_shape()[3].value
        kernel_shape = [1, 1, in_channel, out_channel]

        Kernel[name] = get_kernel(name, kernel_shape, trainable = trainable)
        Bias[name] = get_bias(name, [out_channel], trainable = trainable)

        conv = tf.nn.conv2d(in_data, Kernel[name], strides = [1,1,1,1], padding = 'SAME')
        conv = conv + Bias[name]
        return conv

#====================contrast=====================
def get_deconv_kernel(shape):
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
    return tf.get_variable(name = "deconv_kernel", initializer = init, shape = shape, trainable = False)

def get_deconv_layer(in_data, name, shape, kernel_size = 4, stride = 2):
    with tf.variable_scope(name) as scope:
        strides = [1, stride, stride, 1]
        output_shape = shape

        kernel_shape = [kernel_size, kernel_size, 64, 64]
        Kernel[name] = get_deconv_kernel(kernel_shape)

        deconv = tf.nn.conv2d_transpose(in_data, Kernel[name], output_shape, strides = strides, padding = "SAME")
        return deconv
#=========================

def activateFunc(x):
    return tf.nn.relu(x)
#=============================== get_data ===============================================


recovery_path = './net_weights.npy'
data = {}
start_num = 0
if os.path.isfile(recovery_path):
    data = np.load(recovery_path).item()
    if "num" in data:
        start_num = data["num"] + 1
    else:
        start_num = 0
    print("Recover from " + recovery_path + ". At data " + str(start_num))

   
#===================================== build ================================================

in_data = tf.placeholder("float")
labels = tf.placeholder("float")
# keep_prob = tf.placeholder("float")

# VGG_MEAN = [103.939, 116.779, 123.68]
# red, green, blue = tf.split(in_data, 3, 3)
# bgr = tf.concat([
#                 blue - VGG_MEAN[0],
#                 green - VGG_MEAN[1],
#                 red - VGG_MEAN[2],
#             ], 3)

encoder_conv1_1 = conv_layer(in_data, "conv1_1", kernel_shape = (7,7,3,64), trainable = False)
encoder_conv1_1 = activateFunc(encoder_conv1_1)
encoder_conv1_2 = conv_layer(encoder_conv1_1, "conv1_2", trainable = False)
encoder_conv1_2 = activateFunc(encoder_conv1_2)
pool1, pool1_argmax = max_pool_2x2(encoder_conv1_2, "pool1")

encoder_conv2_1 = conv_layer(pool1, "conv2_1", trainable = True)
encoder_conv2_1 = activateFunc(encoder_conv2_1)
encoder_conv2_2 = conv_layer(encoder_conv2_1, "conv2_2", trainable = True)
encoder_conv2_2 = activateFunc(encoder_conv2_2)
pool2, pool2_argmax = max_pool_2x2(encoder_conv2_2, "pool2")

uppooling_2 = uppooling_layer(pool2, pool2_argmax, tf.shape(encoder_conv2_2))
decoder_conv2_1 = conv_layer(uppooling_2, "decoder_conv2_1", trainable = True)
decoder_conv2_2 = conv_layer(decoder_conv2_1, "decoder_conv2_2", trainable = True)

uppooling_1 = uppooling_layer(decoder_conv2_2, pool1_argmax, tf.shape(encoder_conv1_2))
#uppooling_1 = get_deconv_layer(pool1, name = "uppooling_1", shape = tf.shape(encoder_conv1_2), kernel_size = 4, stride = 2)

decoder_conv1_1 = conv_layer(uppooling_1, "decoder_conv1_1", trainable = False)
decoder_conv1_2 = conv_layer(decoder_conv1_1, "decoder_conv1_2", trainable = False)

#I don't know why I can't fix the score layer
output = channel_change(decoder_conv1_2, "score", trainable = True)

output_labels = tf.nn.softmax(output)

class_balancing = np.array([ 0.00059277,  0.06298457,  0.15726213,  0.05391878,  0.07563882,  0.07698411,
  0.02624592,  0.03258828,  0.01731748,  0.04045499,  0.05523529,  0.03563016,
  0.02754067,  0.05003448,  0.04088812,  0.00969707,  0.07170509,  0.05242883,
  0.03214174,  0.0292276,   0.05148307])
cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(output_labels + 1e-10), class_balancing), axis = 3)
#cross_entropy = -tf.reduce_sum(labels * tf.log(output_labels + 1e-10), axis = 3)
cross_entropy = tf.reduce_mean(cross_entropy)

train_step = tf.train.RMSPropOptimizer(learning_rate = 0.001, decay = 0.9).minimize(cross_entropy)

#=================================== run ==================================================


with tf.Session() as sess:    
    init = tf.global_variables_initializer()
    sess.run(init)

    train_dir = "../train_2012/"
    label_dir = "../label_2012/"
    index_file = "../../VOC2012/ImageSets/Segmentation/train.txt"

    output_tensors = [train_step, cross_entropy, output_labels]
    for key in Kernel:
        output_tensors.append(Kernel[key])
    for key in Bias:
        output_tensors.append(Bias[key])
    

    data_list = utils.get_data_list(index_file)
    l = len(data_list)
    
    for i in range(start_num, l):
        if not os.path.isfile(train_dir+ data_list[i] +'.npy'):
            continue
        if not os.path.isfile(label_dir+ data_list[i] +'.npy'):
            continue
        img = np.load(train_dir+ data_list[i] +'.npy')
        label = np.load(label_dir+ data_list[i] +'.npy')
        
        feed_dict = {in_data: [img], labels:[label]}
        output_values = sess.run(output_tensors, feed_dict = feed_dict)
        print(str(i) + ' ' + str(output_values[1]))
        print(np.round(output_values[2][0,0,0,:], decimals = 3))
        shape = np.shape(img)
        print(np.round(output_values[2][0,int(shape[0]/2),int(shape[1]/2),:], decimals = 3))
        print(np.argmax(output_values[2][0,0,0,:]))
        print(np.argmax(output_values[2][0,int(shape[0]/2),int(shape[1]/2),:]))
        print("========================================")
        
        j = 3
        for key in Kernel:
            data[key] = [output_values[j]]
            j+=1
        for key in Bias:
            data[key].append(output_values[j])
            j+=1
        data["num"] = i
        if (i%20 == 0) or i == l-1:
            np.save(recovery_path, data)
            
        if (i%20 == 0):
            ans_img = utils.change_label_to_img(output_values[2])
            mpimg.imsave("test.jpg", ans_img)
            ans_img = utils.change_label_to_img(np.array([label]))
            mpimg.imsave("label.jpg", ans_img)




