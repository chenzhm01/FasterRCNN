import tensorflow as tf

class SecondStageMinibatch(object):
    def __init__(self,
                 anchors,
                 gtboxes,
                 labels,
                 num_classes,
                 batchsize,
                 positive_rate,
                 p_iou,
                 n_iou):
        self.anchors = anchors
        self.gtboxes = gtboxes

        self.labels = labels
        self.num_classes = num_classes
        self.batchsize = batchsize
        self.positive_rate = positive_rate
        self.p_iou = p_iou
        self.n_iou = n_iou

    def make_minibatch(self):
        with tf.variable_scope('minibatch'):
            anchor_matched_gtboxes, anchor_matched_labels, object_mask = self._pn_samples()
            
            positive_indices = tf.reshape(tf.where(tf.equal(object_mask, 1.0)), [-1])
            num_of_positives = tf.minimum(tf.shape(positive_indices)[0],tf.cast(self.batchsize*self.positive_rate, tf.int32))
            positive_indices = tf.random_shuffle(positive_indices)
            positive_indices = tf.slice(positive_indices,begin=[0],size=[num_of_positives])
        
            negatives_indices = tf.reshape(tf.where(tf.equal(object_mask, 0.0)), [-1])
            num_of_negatives = tf.minimum(tf.cast(self.batchsize, tf.int32)-num_of_positives,tf.shape(negatives_indices)[0])
            negatives_indices = tf.random_shuffle(negatives_indices)
            negatives_indices = tf.slice(negatives_indices, begin=[0], size=[num_of_negatives])

            minibatch_indices = tf.concat([positive_indices, negatives_indices], axis=0)
            minibatch_indices = tf.random_shuffle(minibatch_indices)

            minibatch_anchor_matched_gtboxes = tf.gather(anchor_matched_gtboxes, minibatch_indices)
            minibatch_object_mask = tf.gather(object_mask, minibatch_indices)

            labels = tf.cast(tf.gather(anchor_matched_labels, minibatch_indices), tf.int32)
            labels_one_hot = tf.one_hot(labels, depth=self.num_classes+1)
        return minibatch_indices, minibatch_anchor_matched_gtboxes, minibatch_object_mask, labels_one_hot

    def _pn_samples(self):
        with tf.variable_scope('rpn_find_positive_negative_samples'):
            ious = self.iou_calculate()
            max_iou_each_row = tf.reduce_max(ious, axis=1)
            positives = tf.cast(tf.greater_equal(max_iou_each_row, self.p_iou), tf.int32)

            #each anchor matchs the gtbox
            matchs = tf.cast(tf.argmax(ious, axis=1), tf.int32)
            anchors_matched_gtboxes = tf.gather(self.gtboxes, matchs)

            object_mask = tf.cast(positives, tf.float32)
            anchors_matched_label = tf.gather(self.labels, matchs)
            anchors_matched_label = tf.cast(anchors_matched_label, tf.int32) * positives
        return anchors_matched_gtboxes, anchors_matched_label, object_mask

    def iou_calculate(self):
        with tf.name_scope('iou_caculate'):
            ymin_1, xmin_1, ymax_1, xmax_1 = tf.split(self.anchors, 4, axis=1) #(?,1)
            ymin_2, xmin_2, ymax_2, xmax_2 = tf.unstack(self.gtboxes, 4, axis=1) #(?)
            max_xmin = tf.maximum(xmin_1, xmin_2) #(?, ?)
            min_xmax = tf.minimum(xmax_1, xmax_2)
            max_ymin = tf.maximum(ymin_1, ymin_2)
            min_ymax = tf.minimum(ymax_1, ymax_2)
            overlap_h = tf.maximum(0., min_ymax - max_ymin)
            overlap_w = tf.maximum(0., min_xmax - max_xmin)
            overlaps = overlap_h * overlap_w
            area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)
            area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)
            iou = overlaps / (area_1 + area_2 - overlaps) #[num_anchors, num_gtboxes]
            return iou
        
     