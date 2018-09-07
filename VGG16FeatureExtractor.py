#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 08:43:07 2018

@author: commaai02
"""

import tensorflow as  tf
from nets import vgg
slim = tf.contrib.slim

class VGG16FeatureExtractor(object):
    def __init__(self,
                 _input,
                 is_training):
        self.input = _input
        self.is_training = is_training
    
    def _processor(self):
        VGG_MEAN_rgb = [123.68, 116.779, 103.939]
        image = self.input - VGG_MEAN_rgb
        return image
    
    def get_feature_map(self):
        input_image = self._processor()
        net, endpoints = vgg.vgg_16(input_image, 
                                    num_classes=None, 
                                    is_training=self.is_training)
        feature_map = endpoints['vgg_16/conv5/conv5_3']
        return feature_map
    
    def get_second_stage_feature(self, proposals):
        """
        Args:
            proposals: [num_proposals, 14, 14, num_chanels]
        Return:
            rois: [num_proposals, 4096]
        """
        net = proposals
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.flatten(net)
        net = slim.fully_connected(net, 4096, scope='fc_1')
        net = slim.dropout(net, keep_prob=0.5, is_training=self.is_training, scope='dropout_1')
        net = slim.fully_connected(net, 4096, scope='fc_2')
        net = slim.dropout(net, keep_prob=0.5, is_training=self.is_training, scope='dropout_2')
        return net
    
    
    
    
    