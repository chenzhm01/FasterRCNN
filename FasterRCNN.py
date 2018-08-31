#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 17:27:31 2018

@author: chenzhm
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import vgg
import anchor_utils
from First_Stage_Minibatch import FirstStageMinibatch
from Second_Stage_Minibatch import SecondStageMinibatch


slim = tf.contrib.slim

'''
---------------input-----------------------------
input_dict['image']
input_dict['groundtruth_boxes']
input_dict['groundtruth_classes']
input_dict['image_height']
input_dict['image_width']

-------------build model-------------------------
pred_dict['feature_map']
pred_dict['anchors']

pred_dict['first_stage_box_pred']
pred_dict['first_stage_cls_score']
pred_dict['first_stage_proposals_boxes']
pred_dict['first_stage_proposals_scores']

pred_dict['second_stage_cls_score']
pred_dict['second_stage_box_pred']
pred_dict['second_stage_proposals_scores']
pred_dict['second_stage_proposals_boxes']
pred_dict['second_stage_proposals_classes']

-------------build loss--------------------------
loss_dict['first_stage_box_loss']
loss_dict['first_stage_cls_loss']
loss_dict['second_stage_box_loss']
loss_dict['second_stage_cls_loss']

-----------method--------------------------------
self.FeatureExtractor()
self.first_stage_pred()
self.first_stage_proposals()
self.second_stage_pred()
self.second_stage_proposals()

self.first_stage_losses()
self.second_stage_losses()
self.add_box_image_summaries

'''
import config as cfg

class FasterRCNN(object):
    def __init__(self,
                 input_dict,
                dropout_keep_prob=0.5,
                is_training=True):
        self.input_dict = input_dict
        self.num_classes = cfg.num_classes
        self.anchor_scales = cfg.anchor_scales
        self.anchor_aspect = cfg.anchor_aspect
        self.num_anchors = len(cfg.anchor_scales)*len(cfg.anchor_aspect)
        self.dropout_keep_prob = dropout_keep_prob
        self.is_training = is_training
        self.pred_dict = {}
        self.loss_dict = {}
        self.build_model()

    def build_model(self):
        self.FeatureExtractor()
        self.first_stage_pred()
        self.first_stage_proposals()
        self.second_stage_pred()
        self.second_stage_proposals()
    
    def build_loss(self):
        self.first_stage_losses()
        self.second_stage_losses()
        total_loss = self.loss_dict['first_stage_box_loss']+ \
                     self.loss_dict['first_stage_cls_loss']+ \
                     self.loss_dict['second_stage_box_loss']+ \
                     self.loss_dict['second_stage_cls_loss']
        tf.summary.scalar('total_loss', total_loss)
    
    def FeatureExtractor(self):
        net, endpoints = vgg.vgg_16(self.input_dict['image'], 
                                    num_classes=None, 
                                    is_training=self.is_training)
        feature_map = endpoints['vgg_16/conv5/conv5_3']
        self.pred_dict['feature_map'] = feature_map
        h, w = tf.shape(feature_map)[1], tf.shape(feature_map)[2]
        anchors = anchor_utils.make_anchors(h, w, 256, self.anchor_scales, self.anchor_aspect, stride=16)
        self.pred_dict['anchors'] = anchors
        return feature_map
    
    def first_stage_pred(self):
        with tf.variable_scope('rpn_net'):
            net = slim.conv2d(self.pred_dict['feature_map'], 512, [3, 3], scope='conv_rpn')
            box_pred = slim.conv2d(net, self.num_anchors*4, [1, 1], scope='box_pred', activation_fn=None)
            cls_score = slim.conv2d(net, self.num_anchors*2, [1, 1], scope='cls_score', activation_fn=None)
            first_stage_box_pred = tf.reshape(box_pred, [-1, 4])
            first_stage_cls_score = tf.reshape(cls_score, [-1, 2])
            self.pred_dict['first_stage_box_pred'] = first_stage_box_pred
            self.pred_dict['first_stage_cls_score'] = first_stage_cls_score
            return self.pred_dict
        
    def first_stage_proposals(self):
        with tf.variable_scope('rpn_proposals'):
            bbox_pred = anchor_utils.decode_boxes(self.pred_dict['first_stage_box_pred'], self.pred_dict['anchors'])
            obj_scores = slim.softmax(self.pred_dict['first_stage_cls_score'])[:, 1]            
            obj_scores, top_k_indices = tf.nn.top_k(obj_scores, k=cfg.first_stage_top_k_mns)
            bbox_pred = tf.gather(bbox_pred, top_k_indices)
      
            nms_index = tf.image.non_max_suppression(bbox_pred, 
                                                     obj_scores, 
                                                     cfg.first_stage_max_proposals, 
                                                     cfg.first_stage_nms_iou_threshold)
            valid_boxes = tf.gather(bbox_pred, nms_index)
            if not self.is_training:
                valid_boxes = tf.clip_by_value(valid_boxes, 0.0001, 0.9999)
            valid_scores = tf.gather(obj_scores, nms_index)
            def padd_boxes_with_zeros(boxes, scores, max_num_of_boxes):
                pad_num = tf.cast(max_num_of_boxes, tf.int32) - tf.shape(boxes)[0]
                zero_boxes = tf.zeros(shape=[pad_num, 4], dtype=boxes.dtype)
                zero_scores = tf.zeros(shape=[pad_num], dtype=scores.dtype)
                final_boxes = tf.concat([boxes, zero_boxes], axis=0)
                final_scores = tf.concat([scores, zero_scores], axis=0)
                return final_boxes, final_scores
          
            rpn_proposals_boxes, rpn_proposals_scores = tf.cond(
                tf.less(tf.shape(valid_boxes)[0], cfg.first_stage_max_proposals),
                lambda: padd_boxes_with_zeros(valid_boxes, valid_scores, cfg.first_stage_max_proposals),
                lambda: (valid_boxes, valid_scores))
            
            self.pred_dict['first_stage_proposals_boxes']=rpn_proposals_boxes
            self.pred_dict['first_stage_proposals_scores']=rpn_proposals_scores
            return self.pred_dict

    def second_stage_pred(self):
        with tf.variable_scope('second_stage_pred'):
          bboxes = tf.stop_gradient(self.pred_dict['first_stage_proposals_boxes'])
          rois = tf.image.crop_and_resize(self.pred_dict['feature_map'], 
                                        bboxes, 
                                        tf.zeros(shape=[tf.shape(self.pred_dict['first_stage_proposals_boxes'])[0], ],dtype=tf.int32),
                                        [14, 14])
          rois = slim.max_pool2d(rois, [2, 2], scope='pool5')
          rois = slim.flatten(rois)
          with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.0001)):
            net = slim.fully_connected(rois, 4096, scope='fc_1')
            net = slim.dropout(net, keep_prob=self.dropout_keep_prob, is_training=self.is_training, scope='dropout_1')
            net = slim.fully_connected(net, 4096, scope='fc_2')
            net = slim.dropout(net, keep_prob=self.dropout_keep_prob, is_training=self.is_training, scope='dropout_2')
        
            cls_score = slim.fully_connected(net, self.num_classes+1, activation_fn=None, scope='classifier')
            box_pred = slim.fully_connected(net, self.num_classes*4, activation_fn=None, scope='regressor')
            self.pred_dict['second_stage_cls_score']=cls_score
            self.pred_dict['second_stage_box_pred']=box_pred
            return self.pred_dict
        
    def second_stage_proposals(self):
        with tf.variable_scope('second_stage_proposals'):
            scores = slim.softmax(self.pred_dict['second_stage_cls_score'])
            boxes = tf.reshape(self.pred_dict['second_stage_box_pred'], [-1, 4])
            
            anchor = tf.tile(self.pred_dict['first_stage_proposals_boxes'], [1, self.num_classes])  # [N, 4*num_classes]
            anchor = tf.reshape(anchor, [-1, 4])
            
            decode_boxes = anchor_utils.decode_boxes(boxes, anchor)
            decode_boxes = tf.clip_by_value(decode_boxes, 0.0001, 0.9999)
            decode_boxes = tf.reshape(decode_boxes, [-1, self.num_classes*4])
            
            category = tf.argmax(scores, axis=1)
            object_mask = tf.cast(tf.not_equal(category, 0), tf.float32)

            decode_boxes = decode_boxes * tf.expand_dims(object_mask, axis=1)  # make background box is [0 0 0 0]
            scores = scores * tf.expand_dims(object_mask, axis=1)

            decode_boxes = tf.reshape(decode_boxes, [-1, self.num_classes, 4])  # [N, num_classes, 4]

            decode_boxes_list = tf.unstack(decode_boxes, axis=1)
            score_list = tf.unstack(scores[:, 1:], axis=1)
            
            nms_boxes = []
            nms_scores = []
            category_list = []
            for decode_boxes_i, softmax_scores_i in zip(decode_boxes_list, score_list):
                valid_index = tf.image.non_max_suppression(decode_boxes_i,
                                                           softmax_scores_i,
                                                           cfg.second_max_detections_per_class,
                                                           cfg.second_stage_nms_iou_threshold)
                nms_boxes.append(tf.gather(decode_boxes_i, valid_index))
                nms_scores.append(tf.gather(softmax_scores_i, valid_index))
                category_list.append(tf.gather(category, valid_index))

            all_nms_boxes = tf.concat(nms_boxes, axis=0)
            all_nms_scores = tf.concat(nms_scores, axis=0)
            all_category = tf.concat(category_list, axis=0)

            proposals_num = tf.cond(tf.less(tf.shape(all_nms_boxes)[0], cfg.second_max_total_detections),
                                    lambda: tf.shape(all_nms_boxes)[0],
                                    lambda: cfg.second_max_total_detections)
            _, top_k_indices = tf.nn.top_k(all_nms_scores, k=proposals_num)

            all_nms_boxes = tf.gather(all_nms_boxes, top_k_indices)
            all_nms_scores = tf.gather(all_nms_scores, top_k_indices)
            all_category = tf.gather(all_category, top_k_indices)
            self.pred_dict['second_stage_proposals_scores'] = all_nms_scores
            self.pred_dict['second_stage_proposals_boxes'] = all_nms_boxes
            self.pred_dict['second_stage_proposals_classes'] = all_category
            return self.pred_dict
        
    def first_stage_losses(self):
      with tf.variable_scope('first_stage_losses'):
        minibatch_indices, minibatch_anchor_matched_gtboxes, minibatch_object_mask,\
        minibatch_labels_one_hot = FirstStageMinibatch(self.pred_dict['anchors'],
                                                       self.input_dict['groundtruth_boxes'], 
                                                       batchsize=cfg.first_stage_minibatch_size,
                                                       positive_rate = cfg.first_stage_positive_rate,
                                                       p_iou=cfg.first_stage_positive_threshold,
                                                       n_iou=cfg.first_stage_negative_threshold).make_minibatch()
        
        minibatch_anchors = tf.gather(self.pred_dict['anchors'], minibatch_indices)
        
        num_positive = tf.reduce_sum(minibatch_object_mask)/cfg.first_stage_minibatch_size
        tf.summary.scalar('first_stage_positives', num_positive)
        
        minibatch_encode_gtboxes = anchor_utils.encode_boxes(minibatch_anchor_matched_gtboxes, minibatch_anchors)
    
        minibatch_bbox_pred = tf.gather(self.pred_dict['first_stage_box_pred'], minibatch_indices)
        minibatch_cls_score = tf.gather(self.pred_dict['first_stage_cls_score'], minibatch_indices)
        
        with tf.variable_scope('first_stage_box_loss'):
            diff = minibatch_bbox_pred - minibatch_encode_gtboxes
            abs_diff = tf.cast(tf.abs(diff), tf.float32)
            anchorwise_smooth_l1norm = tf.reduce_sum(tf.where(tf.less(abs_diff, 1),
                                                              0.5*tf.square(abs_diff),abs_diff-0.5),axis=1)
            location_loss = tf.reduce_mean(anchorwise_smooth_l1norm*minibatch_object_mask, axis=0)*cfg.first_stage_box_loss_weight
            slim.losses.add_loss(location_loss)
            tf.summary.scalar('first_stage_box_loss', location_loss)
            self.loss_dict['first_stage_box_loss'] = location_loss
        with tf.variable_scope('first_stage_cls_loss'):
            classification_loss = slim.losses.softmax_cross_entropy(logits=minibatch_cls_score,
                                                                    onehot_labels=minibatch_labels_one_hot)*cfg.first_stage_cls_loss_weight
            tf.summary.scalar('first_stage_cls_loss', classification_loss)
            self.loss_dict['first_stage_cls_loss'] = classification_loss
        return self.loss_dict
    
    def second_stage_losses(self):
      with tf.variable_scope('second_stage_losses'):
        minibatch_indices, minibatch_roiboxes_matched_gtboxes, minibatch_object_mask,\
        minibatch_labels_one_hot = SecondStageMinibatch(self.pred_dict['first_stage_proposals_boxes'],
                                                        self.input_dict['groundtruth_boxes'], 
                                                        self.input_dict['groundtruth_classes'],
                                                        num_classes=self.num_classes,
                                                        batchsize=cfg.second_stage_minibatch_size,
                                                        positive_rate = cfg.second_stage_positive_rate,
                                                        p_iou=cfg.second_stage_positive_threshold,
                                                        n_iou=cfg.second_stage_negative_threshold).make_minibatch()
        
        num_positive = tf.reduce_sum(minibatch_object_mask)/cfg.second_stage_minibatch_size
        tf.summary.scalar('second_stage_positives', num_positive)
        
        minibatch_roiboxes = tf.gather(self.pred_dict['first_stage_proposals_boxes'], minibatch_indices) #[64, 4]
        minibatch_encode_gtboxes = anchor_utils.encode_boxes(minibatch_roiboxes_matched_gtboxes, minibatch_roiboxes) #[64, 4]
        minibatch_encode_gtboxes = tf.tile(minibatch_encode_gtboxes, [1, self.num_classes]) #[64, 4*num_classes]
        
        minibatch_bbox_pred = tf.gather(self.pred_dict['second_stage_box_pred'], minibatch_indices) #[64, 4*num_classes]
        minibatch_cls_score = tf.gather(self.pred_dict['second_stage_cls_score'], minibatch_indices) #[64, 2*num_classes]
        
        cls_weights_list = []
        category_list = tf.unstack(minibatch_labels_one_hot, axis=1)
        for i in range(1, self.num_classes+1):
            _cls_weights = tf.ones(shape=[tf.shape(minibatch_bbox_pred)[0], 4], dtype=tf.float32)
            _cls_weights = _cls_weights * tf.expand_dims(category_list[i], axis=1)
            cls_weights_list.append(_cls_weights)
        cls_weights = tf.concat(cls_weights_list, axis=1)  # [64, num_classes*4]
        
        with tf.variable_scope('second_stage_box_loss'):
            diff = minibatch_bbox_pred - minibatch_encode_gtboxes
            abs_diff = tf.cast(tf.abs(diff), tf.float32)
            anchorwise_smooth_l1norm = tf.reduce_sum(tf.where(tf.less(abs_diff, 1),
                                                              0.5*tf.square(abs_diff)*cls_weights,
                                                              (abs_diff-0.5)*cls_weights),axis=1)
            location_loss = tf.reduce_mean(anchorwise_smooth_l1norm*minibatch_object_mask, axis=0)*cfg.second_stage_box_loss_weight
            slim.losses.add_loss(location_loss)
            tf.summary.scalar('second_stage_box_loss', location_loss)
            self.loss_dict['second_stage_box_loss'] = location_loss
        with tf.variable_scope('second_stage_cls_loss'):
            classification_loss = slim.losses.softmax_cross_entropy(logits=minibatch_cls_score,
                                                                    onehot_labels=minibatch_labels_one_hot)*cfg.second_stage_cls_loss_weight
            tf.summary.scalar('second_stage_cls_loss', classification_loss)
            self.loss_dict['second_stage_cls_loss'] = classification_loss
        return self.loss_dict
    
    def add_box_image_summaries(self):
        first_stage_proposals_boxes = self.pred_dict['first_stage_proposals_boxes']
        first_stage_proposals_scores = self.pred_dict['first_stage_proposals_scores']
        tf.summary.scalar('first_stage_proposals_scores', tf.reduce_max(first_stage_proposals_scores))
        _, top_k_indices_1 = tf.nn.top_k(first_stage_proposals_scores, k=10)
        box_1 = tf.expand_dims(tf.gather(first_stage_proposals_boxes, top_k_indices_1), axis=0)
        image_1 = tf.image.draw_bounding_boxes(self.input_dict['image'], box_1, name='draw_first_stage_proposals_boxes')
        slim.summaries.add_image_summary(image_1, name='first_stage_proposals_boxes')
        
        second_stage_proposals_boxes = self.pred_dict['second_stage_proposals_boxes']
        second_stage_proposals_scores = self.pred_dict['second_stage_proposals_scores']
        tf.summary.scalar('second_stage_proposals_scores', tf.reduce_max(second_stage_proposals_scores))
        _, top_k_indices_2 = tf.nn.top_k(second_stage_proposals_scores, k=10)
        box_2 = tf.expand_dims(tf.gather(second_stage_proposals_boxes, top_k_indices_2), axis=0)
        image_2 = tf.image.draw_bounding_boxes(self.input_dict['image'], box_2, name='draw_second_stage_proposals_boxes')
        slim.summaries.add_image_summary(image_2, name='second_stage_proposals_boxes')
        
    
    