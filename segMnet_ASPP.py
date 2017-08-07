#!/usr/bin/python
#coding=utf-8
#########################################################################
# File Name: segMnet_ASPP.py
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

def conv2d(x, input_filters, output_filters, kernel, strides):
    with tf.variable_scope('conv') as scope:
        shape = [kernel, kernel, input_filters, output_filters]
        initializer = tf.contrib.layers.xavier_initializer()
        weight = tf.Variable(initializer(shape))
        conv = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding='SAME')
        return conv

def atrous_conv2d(x, input_filters, output_filters, kernel, dilation):
    with tf.variable_scope('atrous') as scope:
        shape = [kernel, kernel, input_filters, output_filters]
        initializer = tf.contrib.layers.xavier_initializer()
        weight = tf.Variable(initializer(shape))
        atrous_conv = tf.nn.atrous_conv2d(x, weight, dilation, padding='SAME')
        return atrous_conv

def batch_norm(x, training=True):
    return tf.layers.batch_normalization(x, training=training)

def segMnet(inputs,multiplier):
    segMnet, end_points = mobilenet_v1.mobilenet_v1_base(inputs,depth_multiplier=multiplier)
    
    filters = int(1024*multiplier)

    with tf.variable_scope('gobal_average_pool'):
        shape = tf.shape(segMnet)
        gap = tf.layers.average_pooling2d(segMnet,[shape[1], shape[2]],[shape[1], shape[2]])
    with tf.variable_scope('gap_conv'):
        gconv = batch_norm(conv2d(gap, filters, filters, 1, 1))
    with tf.variable_scope('gap_resize'):
        shape = tf.shape(segMnet)
        gresize = tf.image.resize_bilinear(gconv, [shape[1], shape[2]])

    with tf.variable_scope('atrous0'):
        atnet0 = batch_norm(conv2d(segMnet, filters, filters, 1, 1))
    with tf.variable_scope('atrous1'):
        atnet1 = batch_norm(atrous_conv2d(segMnet, filters, filters, 3, 6))
    with tf.variable_scope('atrous2'):
        atnet2 = batch_norm(atrous_conv2d(segMnet, filters, filters, 3, 12))
    with tf.variable_scope('atrous3'):
        atnet3 = batch_norm(atrous_conv2d(segMnet, filters, filters, 3, 18))

    with tf.variable_scope('concat'):
        segMnet = tf.concat([gresize, atnet0, atnet1, atnet2, atnet3], 3)

    with tf.variable_scope('combine_conv'):
        segMnet = batch_norm(conv2d(segMnet, filters*5, filters, 1, 1))
    #with tf.variable_scope('final_conv'):
    #    segMnet = conv2d(segMnet, filters, 2, 1, 1)

    with tf.variable_scope('deconv'):
        wshape = [64, 64, 2, filters]
        #wshape = [64, 64, 2, 2]
        strides = [1, 32, 32, 1]
        initializer = tf.contrib.layers.xavier_initializer()
        weight = tf.Variable(initializer(wshape))
        shape = tf.shape(inputs)
        output_shape = tf.stack([shape[0], shape[1], shape[2], 2])
        segMnet = tf.nn.conv2d_transpose(segMnet, weight, output_shape, strides=strides, padding='SAME', name='conv_transpose')
    
    return segMnet
