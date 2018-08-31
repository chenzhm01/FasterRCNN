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
from utils import load_step1, step1_result_process

box_pb_path='/home/chenzhm/faster-rcnn/lib3/output/model.pb-9026'

def main():
    g1=tf.Graph()
    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess1 = tf.Session(config=config, graph=g1)
    
    step1_inputs,step1_boxes,step1_classes,step1_scores = load_step1(g1, box_pb_path,sess1)
    
    cap = cv2.VideoCapture(-1)
    cap.set(3,800)
    cap.set(4,600)
    imagedir_for_test = '/home/chenzhm/faster-rcnn/lib3/600c'
    imagelist = os.listdir(imagedir_for_test)
    np.random.shuffle(imagelist)
    for step in range(10000):
    #while(1):
        detect_dict = {}
        st = time.time()

        detect_dict['image_bgr'] = cv2.imread(os.path.join(imagedir_for_test, imagelist[step]))
        #detect_dict['image_bgr'] = cv2.imread('/media/commaai02/disk_1TB/huapu_bread/deeplab_sample/ag_data/images/132495_0_201807_cap80a3_2_1_8.jpg')
        detect_dict['image_rgb'] = cv2.cvtColor(detect_dict['image_bgr'],cv2.COLOR_BGR2RGB)
        feed_dict1={step1_inputs: np.expand_dims(detect_dict['image_rgb'], axis=0)}
        detect_dict['step1_boxes'], detect_dict['step1_classes'], detect_dict['step1_scores'] = sess1.run([step1_boxes,step1_classes,step1_scores], feed_dict=feed_dict1)
        
        detect_dict = step1_result_process(detect_dict, thr=0.8)
        num_step1_detect_boxes = len(detect_dict['step1_classes'])

        for i in range(num_step1_detect_boxes):
            box = detect_dict['step1_boxes'][i]
            cv2.rectangle(detect_dict['image_bgr'], (int(box[1]),int(box[0])), (int(box[3]),int(box[2])), (0,0,255) ,2)
            text1 = 'box: %f'%detect_dict['step1_scores'][i]
            cv2.putText(detect_dict['image_bgr'], text1, (int(box[1]),int(box[0])+10), 0, 0.5, (0,0,0),2)

        #cv2.imshow('detector',detect_dict['image_bgr'])
        cv2.imwrite('/home/chenzhm/faster-rcnn/lib3/p/'+imagelist[step], detect_dict['image_bgr'])
        print('time: %f'%(time.time()-st))
        k = cv2.waitKey(10)&0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    main()
    
