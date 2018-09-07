#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:03:45 2018

@author: chenzhm
"""

import tensorflow as tf
from reader import TfExampleDecoder
from reader import resize_image
import os
from FasterRCNN import FasterRCNN
import config as cfg

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

output_path = './output'
fine_tune_path = '/media/commaai02/disk_1TB/ImageNet_model/inception_v2.ckpt'

def train():
    tf.logging.set_verbosity(tf.logging.INFO)
    record_path='/media/commaai02/disk_1TB/er_sha_dao/train.record'
    input_dict=TfExampleDecoder().get_batch(path_to_record=record_path)
    model = FasterRCNN(input_dict=input_dict,
                       is_training=True)
    model.build_loss()
    model.add_box_image_summaries()
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    global_step = slim.create_global_step()
    total_loss = slim.losses.get_total_loss(add_regularization_losses=cfg.add_regularization_losses)
    learning_rate = tf.train.piecewise_constant(global_step, 
                                                cfg.learning_rate_step, 
                                                cfg.learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    summaries.add(tf.summary.scalar('learning_rate', learning_rate))
    train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True, clip_gradient_norm=10.0)
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    
    variables_to_restore = slim.get_variables_to_restore()
    init_fn = slim.assign_from_checkpoint_fn(fine_tune_path, variables_to_restore, ignore_missing_vars=True)
    
    slim.learning.train(train_op=train_op,
                        logdir=output_path, 
                        init_fn=init_fn,
                        summary_op=summary_op,
                        number_of_steps=cfg.max_steps,
                        log_every_n_steps=10,
                        save_summaries_secs=30, 
                        save_interval_secs=300)
    
def frozen():
    from tensorflow.python.framework import graph_util
    ckpt_path = './output/model.ckpt-30000'
    pb_path = './output/model.pb-30000'
    x = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="image_tensor")
    input_dict = {}
    input_dict['image']=resize_image(x)
    model = FasterRCNN(input_dict=input_dict,
                       is_training=False)
    pred_dict = model.pred_dict
    _ = tf.expand_dims(pred_dict['second_stage_proposals_scores'], axis=0, name='detection_scores')
    _ = tf.expand_dims(pred_dict['second_stage_proposals_boxes'], axis=0, name='detection_boxes')
    _ = tf.expand_dims(pred_dict['second_stage_proposals_classes'], axis=0, name='detection_classes')
    
    sess = tf.Session()
    saver = tf.train.Saver()
    initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(initop)
    saver.restore(sess, ckpt_path)
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess,
                                                                 graph_def,
                                                                 ['detection_scores',
                                                                  'detection_boxes',
                                                                  'detection_classes'])
    with tf.gfile.GFile(pb_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    #train()
    frozen()
    
    
    