


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
    print(y_true.shape)
    y_true, y_len = y_true[0], y_true[1]
    y_pred, _ = y_pred[0], y_pred[1]
    y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
    y_len= tf.cast(tf.convert_to_tensor(y_len), tf.int32)
    #print(y_true.shape)
    #print(y_pred.shape)
    #print(y_len.shape)


    def false_fn(y_true,y_pred):
        print('false')
        return tf.constant(0, dtype=tf.float32)

    # b, 36, 5
    def true_fn(y_true, y_pred, batchsize=4, y_len=36):

        # b, 5
        #gt_idxs = tf.math.argmax(y_true, axis=1)
        # b,
        # this is the length of the original cardiac cycle, we need to use this as cutoff point
        #gt_max = tf.cast(tf.reduce_max(gt_idxs, axis=1), tf.int32)
        #print(y_len)
        #print((y_len*-1)+36)
        # b, ones + zeros (36), 5
        # define ones with (y_length, 5)
        # pad first dim with (before,after) (0, 36-y_length) and the 2nd dim with (0,0)
        print('y_len in true_fn: {}'.format(y_len.shape))
        y_len = tf.cast(tf.squeeze(y_len,axis=0), tf.int32)
        print('y_len before mask: {}'.format(y_len.shape))
        """msk = tf.stack([
            tf.pad(
                tf.ones((int(y_len[i]),5)),
                   ((0,36-int(y_len[i])),(0,0)))
            for i in range(batchsize)])"""

        msk = tf.map_fn(lambda x :
                        tf.cast(
                            tf.pad(
                            tf.ones((int(x),5)),
                            ((0,36-int(x)),(0,0)))
                            ,dtype=tf.float32),
                        y_len, dtype=tf.float32)

        temp_pred = y_pred * msk
        temp_gt = y_true * msk

        gt_idx = tf.math.argmax(temp_gt, axis=1)
        pred_idx = tf.math.argmax(temp_pred, axis=1)
        stacked = tf.cast(tf.stack([gt_idx, pred_idx], axis=-1),tf.int32)
        # returns b, 5,
        # sum the error per entity, and calc the mean over the batches
        diffs = tf.map_fn(lambda x: get_min_dist_for_list(x, y_len), stacked, dtype=tf.int32)
        diffs = tf.cast(tf.reduce_sum(diffs, axis=0),tf.float32)
        diffs = tf.reduce_mean(diffs)
        return diffs

    is_training = tf.constant(y_true.shape.as_list()[0] != None, dtype=tf.bool)
    result_value = tf.cond(is_training, lambda: true_fn(y_true,y_pred, batchsize, y_len), lambda:true_fn(y_true,y_pred, batchsize, y_len))
    return result_value


def get_min_dist_for_list(vals, y_len):
    return tf.map_fn(lambda x :get_min_distance(x, y_len),vals, dtype=tf.int32)

def get_min_distance(vals, mod):
    #assert(mod>(tf.reduce_max(a,b))), 'a: {}, b: {}, mod:{}, '.format(a,b,mod)

    decr_counter = tf.constant(0)
    incr_counter = tf.constant(0)
    print(vals.shape)
    smaller = tf.reduce_min(vals, keepdims=True)
    bigger = tf.reduce_max(vals, keepdims=True)

    i1 = bigger
    while (i1 != smaller):
        tf.autograph.experimental.set_loop_options(
        shape_invariants=[(i1, tf.TensorShape([None]))])

        decr_counter = decr_counter + 1
        i1 = i1 - 1

    i1 = bigger
    while (i1 != smaller):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(i1, tf.TensorShape([None]))])
        incr_counter = incr_counter + 1
        i1 = tf.math.mod((i1 + 1), mod)

    return tf.reduce_min(tf.stack([decr_counter, incr_counter]))


def cce_wrapper(y_true,y_pred):
    y_true, y_len = y_true[0], y_true[1]
    y_pred, _ = y_pred[0], y_pred[1]
    return tf.keras.losses.categorical_crossentropy(y_true,y_pred,label_smoothing=0.2)

def mse_wrapper(y_true,y_pred):
    y_true, y_len = tf.unstack(y_true, axis=1)
    y_pred, _ = tf.unstack(y_pred, axis=1)
    return tf.keras.losses.mse(y_true, y_pred)

def ca_wrapper(y_true, y_pred):
    y_true, y_len = tf.unstack(y_true, axis=1)
    y_pred, _ = tf.unstack(y_pred, axis=1)
    return tf.keras.metrics.CategoricalAccuracy()(y_true, y_pred)
