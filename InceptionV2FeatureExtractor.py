#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 09:15:30 2018

@author: commaai02
"""

import tensorflow as  tf
from nets import inception_v2
slim = tf.contrib.slim

class InceptionV2FeatureExtractor(object):
    def __init__(self,
                 _input,
                 is_training):
        self.input = _input
        self.is_training = is_training
    
    def _processor(self):
        image = (2.0/255.0)*self.input - 1.0
        return image
    
    def get_feature_map(self):
        input_image = self._processor()
        bn_params={'is_training':False,
                   'scale':False,
                   'decay':0.9997,
                   'epsilon':0.001}
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=bn_params):
            _, activations = inception_v2.inception_v2_base(input_image,
                                                            final_endpoint='Mixed_4e')
        feature_map = activations['Mixed_4e']
        return feature_map
    
    def get_second_stage_feature(self, proposals):
        """
        Args:
            proposals: [num_proposals, 14, 14, num_chanels]
        Return:
            rois: [num_proposals, 1024]
        """
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                             stride=1,
                             padding='SAME'):
            end_point = 'Mixed_5a'
            net = proposals
            depth = lambda d: max(d, 16)
            concat_dim = 3
            trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(128), [1, 1],
                                           weights_initializer=trunc_normal(0.09),
                                           scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2,
                                           scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1],
                                           weights_initializer=trunc_normal(0.09),
                                           scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(256), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2,
                                           scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2])
    
            end_point = 'Mixed_5b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1],
                                           weights_initializer=trunc_normal(0.09),
                                           scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                           scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1],
                                           weights_initializer=trunc_normal(0.09),
                                           scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                           scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(128), [1, 1],
                                           weights_initializer=trunc_normal(0.1),
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                
            end_point = 'Mixed_5c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1],
                                           weights_initializer=trunc_normal(0.09),
                                           scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                           scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(192), [1, 1],
                                           weights_initializer=trunc_normal(0.09),
                                           scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                           scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(128), [1, 1],
                                           weights_initializer=trunc_normal(0.1),
                                           scope='Conv2d_0b_1x1')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
            
            end_point = 'AvgPool'
            with tf.variable_scope(end_point):
                net = slim.avg_pool2d(net, [7,7], padding='VALID',scope='AvgPool_7x7')
                net = tf.squeeze(net, [1, 2], name='squeezed')
                net = slim.dropout(net, keep_prob=0.8, is_training=self.is_training, scope='Dropout_1b')
        return net
