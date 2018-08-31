import tensorflow as tf

class FirstStageMinibatch(object):
    def __init__(self,
                 anchors,
                 gtboxes,
                 batchsize,
                 positive_rate, 
                 p_iou,
                 n_iou,
                 n_iou_0=0.1):
        self.anchors = anchors
        self.gtboxes = gtboxes
        self.batchsize = batchsize
        self.positive_rate = positive_rate
        self.p_iou = p_iou
        self.n_iou = n_iou
        self.n_iou_0 = n_iou_0

    def make_minibatch(self):
        with tf.variable_scope('minibatch'):
            pn_labels, anchor_matched_gtboxes, object_mask = self._pn_samples()
            positive_indices = tf.reshape(tf.where(tf.equal(pn_labels, 1.0)), [-1])
            num_of_positives = tf.minimum(tf.shape(positive_indices)[0],tf.cast(self.batchsize*self.positive_rate, tf.int32))
            positive_indices = tf.random_shuffle(positive_indices)
            positive_indices = tf.slice(positive_indices,begin=[0],size=[num_of_positives])
        
            negatives_indices = tf.reshape(tf.where(tf.equal(pn_labels, 0.0)), [-1])
            num_of_negatives = tf.minimum(tf.cast(self.batchsize, tf.int32)-num_of_positives,tf.shape(negatives_indices)[0])
            negatives_indices = tf.random_shuffle(negatives_indices)
            negatives_indices = tf.slice(negatives_indices, begin=[0], size=[num_of_negatives])

            minibatch_indices = tf.concat([positive_indices, negatives_indices], axis=0)
            minibatch_indices = tf.random_shuffle(minibatch_indices)

            minibatch_anchor_matched_gtboxes = tf.gather(anchor_matched_gtboxes, minibatch_indices)
            minibatch_object_mask = tf.gather(object_mask, minibatch_indices)
      
            labels = tf.cast(tf.gather(pn_labels, minibatch_indices), tf.int32)
            labels_one_hot = tf.one_hot(labels, depth=2)
        return minibatch_indices, minibatch_anchor_matched_gtboxes, minibatch_object_mask, labels_one_hot

    def _pn_samples(self):
        with tf.variable_scope('rpn_find_positive_negative_samples'):
            ious = self.iou_calculate()
            max_iou_each_row = tf.reduce_max(ious, axis=1)
            max_iou_each_column = tf.reduce_max(ious, 0)
        
            pn_labels = tf.ones(shape=[tf.shape(self.anchors)[0], ], dtype=tf.float32) * (-1)
            positives1 = tf.greater_equal(max_iou_each_row, self.p_iou)
            positives2 = tf.reduce_sum(tf.cast(tf.equal(ious, max_iou_each_column), tf.float32), axis=1)
            positives = tf.logical_or(positives1, tf.cast(positives2, tf.bool))
            pn_labels = pn_labels + 2 * tf.cast(positives, tf.float32)  #positive is 1, ignored and background is -1

            #each anchor matchs the gtbox
            matchs = tf.cast(tf.argmax(ious, axis=1), tf.int32)
            anchors_matched_gtboxes = tf.gather(self.gtboxes, matchs)
        
            negatives = tf.less(max_iou_each_row, self.n_iou)
            negatives = tf.logical_and(negatives, tf.greater_equal(max_iou_each_row, self.n_iou_0))
            pn_labels = pn_labels + tf.cast(negatives, tf.float32)  #[N, ] positive is >=1.0, negative is 0, ignored is -1.0

            _positives = tf.cast(tf.greater_equal(pn_labels, 1.0), tf.float32)
            _ignored = tf.cast(tf.equal(pn_labels, -1.0), tf.float32) * -1
            pn_labels = _positives + _ignored
            object_mask = tf.cast(_positives, tf.float32)
        return pn_labels, anchors_matched_gtboxes, object_mask

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
