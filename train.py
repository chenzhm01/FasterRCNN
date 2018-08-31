#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:03:45 2018

@author: chenzhm
"""

import tensorflow as tf
from reader import TfExampleDecoder

from FasterRCNN import FasterRCNN

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

output_path = './output'
fine_tune_path = '/media/chenzhm/Data/ImageNet_model/vgg_16.ckpt'

max_steps = 50000
num_classes = 1

def train():
    tf.logging.set_verbosity(tf.logging.INFO)
    record_path='/home/chenzhm/faster-rcnn/train.record'
    input_dict=TfExampleDecoder().get_batch(path_to_record=record_path)
    model = FasterRCNN(input_dict=input_dict,
                       dropout_keep_prob=0.5,
                       is_training=True)
    model.build_loss()
    model.add_box_image_summaries()
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    global_step = slim.create_global_step()
    total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
    learning_rate = tf.train.piecewise_constant(global_step, [10000, 12000], [0.001, 0.0001, 0.0001])
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
                        number_of_steps=max_steps,
                        save_summaries_secs=30, 
                        save_interval_secs=300)
    
def frozen():
    from tensorflow.python.framework import graph_util
    ckpt_path = './output/model.ckpt-9026'
    pb_path = './output/model.pb-9026'
    x = tf.placeholder(tf.float32, shape=[1, None, None, 3], name="image_tensor")
    VGG_MEAN_rgb = [123.68, 116.779, 103.939]
    image = x - VGG_MEAN_rgb
    input_dict = {}
    input_dict['image']=image
    model = FasterRCNN(input_dict=input_dict,
                       dropout_keep_prob=1.0,
                       is_training=False)

    _ = tf.expand_dims(model.pred_dict['second_stage_proposals_scores'], axis=0, name='detection_scores')
    _ = tf.expand_dims(model.pred_dict['second_stage_proposals_boxes'], axis=0, name='detection_boxes')
    _ = tf.expand_dims(model.pred_dict['second_stage_proposals_classes'], axis=0, name='detection_classes')
    
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
    #train()
    frozen()
    
    
    