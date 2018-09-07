#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 09:25:58 2018

@author: commaai02
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:48:59 2018

@author: chenzhm
"""

import numpy as np
import tensorflow as tf
import cv2
import time
import os

def load_step1(g, pb_path, sess):
    with sess.as_default():
      with g.as_default():
        gx = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
          serialized_graph = fid.read()
          gx.ParseFromString(serialized_graph)
          tf.import_graph_def(gx, name='')
        step1_inputs = g.get_tensor_by_name('image_tensor:0')
        step1_scores = tf.squeeze(g.get_tensor_by_name('detection_scores:0'), [0])
        step1_boxes = tf.squeeze(g.get_tensor_by_name('detection_boxes:0'), [0])
        step1_classes = tf.squeeze(g.get_tensor_by_name('detection_classes:0'), [0])
        return step1_inputs,step1_boxes,step1_classes,step1_scores

def step1_result_process(detect_dict, thr=0.8):
    h,w,_ = detect_dict['image_rgb'].shape
    area = np.where(detect_dict['step1_scores']>thr)
    detect_dict['step1_scores'] = detect_dict['step1_scores'][area]
    detect_dict['step1_classes'] = detect_dict['step1_classes'][area]
    detect_dict['step1_boxes'] = detect_dict['step1_boxes'][area]
    if len(detect_dict['step1_scores'])>0:
        box = detect_dict['step1_boxes']
        detect_dict['step1_boxes'] = box*[h,w,h,w]
    return detect_dict

box_pb_path='../output/model.pb-30000'

def main():
    g1=tf.Graph()
    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess1 = tf.Session(config=config, graph=g1)
    
    step1_inputs,step1_boxes,step1_classes,step1_scores = load_step1(g1, box_pb_path,sess1)
    
    cap = cv2.VideoCapture(-1)
    cap.set(3,800)
    cap.set(4,600)
    imagedir_for_test = './images'
    imagelist = os.listdir(imagedir_for_test)
    np.random.shuffle(imagelist)
    for step in range(len(imagelist)):
    #while(1):
        detect_dict = {}
        st = time.time()

        detect_dict['image_bgr'] = cv2.imread(os.path.join(imagedir_for_test, imagelist[step]))
        im = cv2.cvtColor(detect_dict['image_bgr'],cv2.COLOR_BGR2RGB)
        detect_dict['image_rgb'] = im
        feed_dict1={step1_inputs: np.expand_dims(detect_dict['image_rgb'], axis=0)}
        detect_dict['step1_boxes'], detect_dict['step1_classes'], detect_dict['step1_scores'] = sess1.run([step1_boxes,step1_classes,step1_scores], feed_dict=feed_dict1)
        
        
        detect_dict = step1_result_process(detect_dict, thr=0.8)
        num_step1_detect_boxes = len(detect_dict['step1_classes'])

        for i in range(num_step1_detect_boxes):
            box = detect_dict['step1_boxes'][i]
            cv2.rectangle(detect_dict['image_bgr'], (int(box[1]),int(box[0])), (int(box[3]),int(box[2])), (0,0,255) ,2)
            text1 = 'box: %f'%detect_dict['step1_scores'][i]
            cv2.putText(detect_dict['image_bgr'], text1, (int(box[1]),int(box[0])+10), 0, 0.5, (0,255,0),2)

        #cv2.imshow('detector',detect_dict['image_bgr'])
        cv2.imwrite('./images/p-'+imagelist[step], detect_dict['image_bgr'])
        print('time: %f'%(time.time()-st))
        k = cv2.waitKey(10)&0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    main()
    
