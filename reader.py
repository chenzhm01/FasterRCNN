#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:57:31 2018

@author: chenzhm
"""

import tensorflow as tf
import config as cfg
slim_example_decoder = tf.contrib.slim.tfexample_decoder

def resize_image(image):
    shape = tf.shape(image)
    if image.get_shape().ndims == 3:
        height = tf.to_float(shape[0])
        width = tf.to_float(shape[1])
    if image.get_shape().ndims == 4:
        height = tf.to_float(shape[1])
        width = tf.to_float(shape[2])
    scale0 = cfg.min_dimension/tf.cast(tf.minimum(height, width), tf.float32)
    scale1 = cfg.max_dimension/tf.cast(tf.maximum(height, width), tf.float32)
    new_height = tf.to_int32(height*scale0)
    new_width = tf.to_int32(width*scale0)
    new_height, new_width = tf.cond(tf.less(tf.maximum(new_height, new_width), 1024),
                                    true_fn=lambda:(new_height, new_width),
                                    false_fn=lambda:(tf.to_int32(height*scale1),
                                                     tf.to_int32(width*scale1)))
    image = tf.image.resize_images(image, (new_height, new_width))
    return image

class TfExampleDecoder(object):
  def __init__(self):
    self.keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, 1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, 1),
        # Object boxes and classes.
        'image/object/bbox/xmin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(tf.int64),
        'image/object/class/text':
            tf.VarLenFeature(tf.string),
        'image/object/area':
            tf.VarLenFeature(tf.float32),
        'image/object/is_crowd':
            tf.VarLenFeature(tf.int64),
        'image/object/difficult':
            tf.VarLenFeature(tf.int64),
        'image/object/group_of':
            tf.VarLenFeature(tf.int64),
        'image/object/weight':
            tf.VarLenFeature(tf.float32),
    }
    self.items_to_handlers = {
        'image':
            slim_example_decoder.Image(
                image_key='image/encoded',
                format_key='image/format',
                channels=3),
        'groundtruth_boxes': (
            slim_example_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'],
                                             'image/object/bbox/')),
        'groundtruth_classes':
            slim_example_decoder.Tensor('image/object/class/label'),
        'image_height':
             slim_example_decoder.Tensor('image/height'),
        'image_width':
             slim_example_decoder.Tensor('image/width'),
    }

  def decode(self, tf_example_string_tensor):
    serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                    self.items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    tensor_dict['image'].set_shape([None, None, 3])
    
    tensor_dict = self.processor(tensor_dict)
    return tensor_dict

  def flip_left_right(self, tensor_dict):
    new_image = tf.image.flip_left_right(tensor_dict['image'])
    ymin, xmin, ymax, xmax = tf.split(tensor_dict['groundtruth_boxes'], 4, axis=1)
    new_groundtruth_boxes = tf.concat((ymin, 1.0-xmax, ymax, 1.0-xmin), axis=1)
    return new_image, new_groundtruth_boxes

  def random_flip_left_right(self, tensor_dict):
      condition = tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5)
      #condition = tf.less(0.1, 0.5)
      new_image, new_groundtruth_boxes = tf.cond(condition,
                                                 lambda: self.flip_left_right(tensor_dict),
                                                 lambda: (tensor_dict['image'], tensor_dict['groundtruth_boxes']))
      tensor_dict['image'] = new_image
      tensor_dict['groundtruth_boxes'] = new_groundtruth_boxes
      return tensor_dict

  def random_brightness(self, tensor_dict):
      image = tf.image.random_brightness(tensor_dict['image'], 0.3)
      image = tf.clip_by_value(image, 0, 255)
      tensor_dict['image'] = image
      return tensor_dict

  def processor(self, tensor_dict):
    tensor_dict['image'] = resize_image(tensor_dict['image'])
    tensor_dict = self.random_flip_left_right(tensor_dict)
    #tensor_dict = self.random_brightness(tensor_dict)
    tensor_dict['image'] = tf.expand_dims(tensor_dict['image'], axis=0)
    return tensor_dict

  def get_batch(self, path_to_record):
    dataset = tf.data.TFRecordDataset(path_to_record)
    dataset = dataset.map(self.decode)
    dataset = dataset.repeat(1000).shuffle(256)
    iterator = dataset.make_one_shot_iterator()
    tensor_dict = iterator.get_next()
    return tensor_dict

if __name__ == '__main__':
    record_path='/home/chenzhm/faster-rcnn/train.record'
    x = TfExampleDecoder().get_batch(path_to_record=record_path)
    sess = tf.Session()
    initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(initop)
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    
    x_ = sess.run([x])
    
    coord.request_stop()
    coord.join(threads)

