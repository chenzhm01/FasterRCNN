#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:52:51 2018

@author: commaai02
"""
import hashlib
import io
import os
import re

from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', './BreadCountReality_V2', 'Root.')
FLAGS = flags.FLAGS

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def recursive_parse_xml_to_dict(xml):
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

def get_label_map_dict(label_map_pbtxt):
    items=open(label_map_pbtxt).read().split('item')
    label_map_dict={}
    for item in items:
        if 'name' in item:
            _name = re.search(r"'(.*)'",item).group(1)
            _id = int(re.search(r"id:(.*)\s",item).group(1))
            label_map_dict[_name]=_id
    return label_map_dict

def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False):

  img_path = os.path.join(image_subdirectory, data['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  for i in range(len(data['object'])):
    obj = data['object'][i]

    xmin = float(obj['bndbox']['xmin'])
    xmax = float(obj['bndbox']['xmax'])
    ymin = float(obj['bndbox']['ymin'])
    ymax = float(obj['bndbox']['ymax'])

    xmins.append(xmin / width)
    ymins.append(ymin / height)
    xmaxs.append(xmax / width)
    ymaxs.append(ymax / height)
    class_name = obj['name']
    #class_name = 'obj'
    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])

  feature_dict = {
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/filename': bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': bytes_feature(key.encode('utf8')),
      'image/encoded': bytes_feature(encoded_jpg),
      'image/format': bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': float_list_feature(xmins),
      'image/object/bbox/xmax': float_list_feature(xmaxs),
      'image/object/bbox/ymin': float_list_feature(ymins),
      'image/object/bbox/ymax': float_list_feature(ymaxs),
      'image/object/class/text': bytes_list_feature(classes_text),
      'image/object/class/label': int64_list_feature(classes),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example

def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):

  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 100 == 0:
      print('On image %d of %d'%(idx, len(examples)))
    xml_path = os.path.join(annotations_dir, example + '.xml')
    if not os.path.exists(xml_path):
      print('Could not find %s, ignoring example.'%xml_path)
      continue
    with tf.gfile.GFile(xml_path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = recursive_parse_xml_to_dict(xml)['annotation']

    try:
      tf_example = dict_to_tf_example(
          data,
          label_map_dict,
          image_dir)
      writer.write(tf_example.SerializeToString())
    except ValueError:
      print('Invalid example: %s, ignoring.'%xml_path)
  writer.close()

def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict = get_label_map_dict(os.path.join(data_dir, 'label_map.pbtxt'))
  image_dir = os.path.join(data_dir, 'JPEGImages')
  annotations_dir = os.path.join(data_dir, 'Annotations')
  examples_path = os.path.join(data_dir, 'examples.txt')  
  train_output_path = os.path.join(data_dir, 'train.record')

  np.random.seed(42)
  examples_list = np.loadtxt(examples_path, dtype=str).tolist()
  np.random.shuffle(examples_list)
  create_tf_record(
      train_output_path,
      label_map_dict,
      annotations_dir,
      image_dir,
      examples_list)

if __name__ == '__main__':
  tf.app.run()
