#!/usr/bin/python
#coding=utf-8
#########################################################################
# File Name: segMnet.py
# Author: daniel.wang
# Mail: wangzhanoop@163.com
# Created Time: Thu 15 Jun 2017 02:53:32 PM CST
# Brief: 
#########################################################################
import numpy as np
import tensorflow as tf
import sys
sys.path.append("~/tensorflow_models/slim/nets")
from nets import mobilenet_v1

slim = tf.contrib.slim

def segMnet(inputs, multiplier):
    segMnet, end_points = mobilenet_v1.mobilenet_v1_base(inputs, depth_multiplier=multiplier)
    filters = int(1024*multiplier)

    with tf.variable_scope('deconv'):
        wshape = [64, 64, 2, filters]
        strides = [1, 32, 32, 1]
        initializer = tf.contrib.layers.xavier_initializer()
        weight = tf.Variable(initializer(wshape))
        shape = tf.shape(inputs)
        output_shape = tf.stack([shape[0], shape[1], shape[2], 2])
        segMnet = tf.nn.conv2d_transpose(segMnet, weight, output_shape, strides=strides, padding='SAME', name='conv_transpose')
    
    return segMnet
