


import numpy as np
import tensorflow as tf


def meandiff( y_true, y_pred, batchsize=4):

    """
    Average over the batches
    the sum of the absolute difference between two arrays
    y_true and y_pred are one-hot vectors with the following shape
    batchsize * timesteps * phase classes
    e.g.: 4 * 36 * 5
    First for gt and pred:
    - get the timesteps per phase with the highest probability
    - get the absolute difference between gt and pred
    (- later we can slice each patient by the max value in the corresponding gt indices)
    - sum the diff per entity
    - calc the mean over all examples

    Parameters
    ----------
    y_true :
    y_pred :

    Returns tf.float32 scalar
    -------

    """
    y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)

    def false_fn(y_true,y_pred,batchsize):
        print('false')
        return tf.constant(0, dtype=tf.float32)

    # b, 36, 5
    def true_fn(y_true, y_pred, batchsize):
        # b, 5
        gt_idxs = tf.math.argmax(y_true, axis=1)
        # b,
        gt_max = tf.cast(tf.reduce_max(gt_idxs, axis=1), tf.int32)
        print(gt_max.shape)
        # b, ones + zeros (36), 5
        msk = tf.stack([
            tf.pad(
                tf.ones((gt_max[i],5)),
                   ((0,36-gt_max[i]),(0,0)))
            for i in range(batchsize)])

        temp_pred = y_pred * msk
        temp_gt = y_true * msk

        gt_idx = tf.math.argmax(temp_gt, axis=1)
        pred_idx = tf.math.argmax(temp_pred, axis=1)
        stacked = tf.stack([gt_idx, pred_idx], axis=-1)
        # returns b, 5,
        # sum the error per entity, and calc the mean over the batches
        #diffs = tf.stack([tf.stack(get_min_dist_for_list(gt_idx[i],pred_idx[i], batchsize)) for i in tf.range(batchsize)])
        diffs = tf.map_fn(lambda x: get_min_dist_for_list(x, batchsize), stacked, dtype=tf.int32)
        diffs = tf.cast(tf.reduce_sum(diffs, axis=0),tf.float32)
        diffs = tf.reduce_mean(diffs)
        return diffs

    is_training = tf.Variable(y_true.shape.as_list()[0] != None, dtype=tf.bool)
    result_value = tf.cond(is_training, lambda: true_fn(y_true,y_pred, batchsize), lambda:false_fn(y_true,y_pred,batchsize))
    return result_value


def get_min_dist_for_list(vals, batchsize):
    length = tf.reduce_max(vals)
    #stacked = tf.stack([lst_a, lst_b], axis=1)
    #return [get_min_distance(lst_a[i],lst_b[i],length) for i in tf.range(batchsize)]
    return tf.map_fn(lambda x :get_min_distance(x, length),vals, dtype=tf.int32)

def get_min_distance(vals, mod):
    #assert(mod>(tf.reduce_max(a,b))), 'a: {}, b: {}, mod:{}, '.format(a,b,mod)

    decr_counter = 0
    incr_counter = 0

    smaller = tf.reduce_min(vals)
    bigger = tf.reduce_max(vals)

    i1 = bigger
    while (i1 != smaller):
        decr_counter = decr_counter + 1
        i1 = i1 - 1

    i1 = bigger
    while (i1 != smaller):
        incr_counter = incr_counter + 1
        i1 = tf.math.mod((i1 + 1), mod)

    return tf.reduce_min(tf.stack([decr_counter, incr_counter]))