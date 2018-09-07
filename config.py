#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:49:18 2018

@author: chenzhm
"""

num_classes = 1
min_dimension = 600.0
max_dimension = 1024.0
anchor_scales = [0.25, 0.5, 1.0, 2.0]
anchor_aspect = [0.5, 1.0, 2.0]

anchor_scale_factors = [10.0, 10.0, 5.0, 5.0]

#model = 'vgg16'
model = 'inception_v2'



add_regularization_losses = False
max_steps = 30000
learning_rate_step = [180000, 200000]
learning_rate = [0.001, 0.0001, 0.00001]
num_stages = 2

first_stage_positive_threshold = 0.7
first_stage_negative_threshold = 0.3
first_stage_positive_rate = 0.5
first_stage_minibatch_size = 128
first_stage_top_k_mns = 3000
first_stage_nms_iou_threshold = 0.7
first_stage_max_proposals = 100
first_stage_box_loss_weight = 2.0
first_stage_cls_loss_weight = 1.0


second_stage_positive_threshold = 0.6
second_stage_negative_threshold = 0.6
second_stage_positive_rate = 0.25
second_stage_minibatch_size = 64
second_max_detections_per_class = 100
second_stage_nms_iou_threshold = 0.6
second_max_total_detections = 300
second_stage_box_loss_weight = 2.0
second_stage_cls_loss_weight = 1.0
