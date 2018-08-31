#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:49:18 2018

@author: chenzhm
"""

num_classes = 1
min_dimension = 600
max_dimension = 1024
anchor_scales = [0.25, 0.5, 1.0, 2.0]
anchor_aspect = [0.5, 1.0, 2.0]

first_stage_positive_threshold = 0.7
first_stage_negative_threshold = 0.3
first_stage_positive_rate = 0.5
first_stage_minibatch_size = 256
first_stage_top_k_mns = 3000
first_stage_nms_iou_threshold = 0.7
first_stage_max_proposals = 300
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
