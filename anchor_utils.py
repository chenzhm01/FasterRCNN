import tensorflow as tf

def make_anchors(h,w, base_anchor_size=256, anchor_scales=[0.5,1,2], anchor_ratios=[0.5,1,2], stride=16, name='make_anchors'):
    with tf.variable_scope(name):
        h = tf.cast(h, tf.float32)
        w = tf.cast(w, tf.float32)
        base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], dtype=tf.float32)
        base_anchors = enum_ratios(enum_scales(base_anchor, anchor_scales), anchor_ratios)
        _, _, ws, hs = tf.unstack(base_anchors, axis=1)
        x_centers = tf.range(w, dtype=tf.float32) * stride
        y_centers = tf.range(h, dtype=tf.float32) * stride
        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)
        ws, x_centers = tf.meshgrid(ws, x_centers)
        hs, y_centers = tf.meshgrid(hs, y_centers)
        box_centers = tf.stack([y_centers/(h*16), x_centers/(w*16)], axis=2)
        box_centers = tf.reshape(box_centers, [-1, 2])
        box_sizes = tf.stack([hs/(h*16), ws/(w*16)], axis=2)
        box_sizes = tf.reshape(box_sizes, [-1, 2])
        _anchors = tf.concat([box_centers-0.5*box_sizes, box_centers+0.5*box_sizes], axis=1)
        final_anchors = tf.reshape(_anchors, [-1, 4])
    return final_anchors

def enum_scales(base_anchor, anchor_scales):
    with tf.variable_scope('enum_scales'):
        anchor_scales = tf.constant(anchor_scales, dtype=tf.float32)
        anchor_scales = tf.reshape(anchor_scales, [-1, 1])
        return base_anchor * anchor_scales
def enum_ratios(anchors, anchor_ratios):
    with tf.variable_scope('enum_ratios'):
        anchor_ratios = tf.constant(anchor_ratios, dtype=tf.float32)
        _, _, hs, ws = tf.unstack(anchors, axis=1)
        sqrt_ratios = tf.sqrt(anchor_ratios)
        sqrt_ratios = tf.expand_dims(sqrt_ratios, axis=1)
        ws = tf.reshape(ws / sqrt_ratios, [-1])
        hs = tf.reshape(hs * sqrt_ratios, [-1])
        num_anchors_per_location = tf.shape(ws)[0]
        return tf.transpose(tf.stack([tf.zeros([num_anchors_per_location, ]),
                                      tf.zeros([num_anchors_per_location,]),
                                      ws, hs]))

def encode_boxes(unencode_boxes, reference_anchors, scale_factors=[10.0, 10.0, 5.0, 5.0]):
    with tf.variable_scope('encode'):
        ymin, xmin, ymax, xmax = tf.unstack(unencode_boxes, axis=1)
        reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_anchors, axis=1)
        x_center = (xmin + xmax) / 2.
        y_center = (ymin + ymax) / 2.
        w = xmax - xmin
        h = ymax - ymin
        reference_xcenter = (reference_xmin + reference_xmax) / 2.
        reference_ycenter = (reference_ymin + reference_ymax) / 2.
        reference_w = reference_xmax - reference_xmin
        reference_h = reference_ymax - reference_ymin

        reference_w += 1e-8
        reference_h += 1e-8
        w += 1e-8
        h += 1e-8

        t_xcenter = (x_center - reference_xcenter) / reference_w
        t_ycenter = (y_center - reference_ycenter) / reference_h
        t_w = tf.log(w / reference_w)
        t_h = tf.log(h / reference_h)
        if scale_factors:
            t_xcenter *= scale_factors[0]
            t_ycenter *= scale_factors[1]
            t_w *= scale_factors[2]
            t_h *= scale_factors[3]
        return tf.transpose(tf.stack([t_ycenter, t_xcenter, t_h, t_w]))

def decode_boxes(encode_boxes, reference_anchors, scale_factors=[10.0, 10.0, 5.0, 5.0]):
    with tf.variable_scope('decode'):
        t_ycenter, t_xcenter, t_h, t_w = tf.unstack(encode_boxes, axis=1)
        if scale_factors:
            t_xcenter /= scale_factors[0]
            t_ycenter /= scale_factors[1]
            t_w /= scale_factors[2]
            t_h /= scale_factors[3]
        reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_anchors, axis=1)

        reference_xcenter = (reference_xmin + reference_xmax) / 2.
        reference_ycenter = (reference_ymin + reference_ymax) / 2.
        reference_w = reference_xmax - reference_xmin
        reference_h = reference_ymax - reference_ymin

        predict_xcenter = t_xcenter * reference_w + reference_xcenter
        predict_ycenter = t_ycenter * reference_h + reference_ycenter
        predict_w = tf.exp(t_w) * reference_w
        predict_h = tf.exp(t_h) * reference_h

        predict_xmin = predict_xcenter - predict_w / 2.
        predict_xmax = predict_xcenter + predict_w / 2.
        predict_ymin = predict_ycenter - predict_h / 2.
        predict_ymax = predict_ycenter + predict_h / 2.
        return tf.transpose(tf.stack([predict_ymin, predict_xmin, predict_ymax, predict_xmax]))