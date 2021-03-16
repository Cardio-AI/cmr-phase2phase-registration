


import numpy as np
import tensorflow as tf


def meandiff( y_true, y_pred, apply_sum=True, apply_average=True, as_loss=False):

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
    #print(y_true.shape)
    y_true, y_len_msk = tf.unstack(y_true,2,axis=1)
    y_pred, _ = tf.unstack(y_pred,2,axis=1)

    y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
    y_len_msk = tf.cast(tf.convert_to_tensor(y_len_msk), tf.float32)

    # b, 36, 5
    temp_pred = y_pred * y_len_msk
    temp_gt = y_true * y_len_msk

    # get the original lengths of each mask in the current batch
    # b, 1
    y_len = tf.cast(tf.reduce_sum(y_len_msk[:,:,0], axis=1),dtype=tf.int32)#

    #print('y_len shape: {}'.format(y_len.shape))
    # returns b, 5,
    gt_idx = tf.cast(tf.math.argmax(temp_gt, axis=1), dtype=tf.int32)
    pred_idx = tf.cast(tf.math.argmax(temp_pred, axis=1), dtype=tf.int32)

    #gt_idx = tf.cast(DifferentiableArgmax(temp_gt, axis=1), dtype=tf.float32)#
    #pred_idx = tf.cast(DifferentiableArgmax(temp_pred, axis=1), dtype=tf.float32)
    filled_length = tf.repeat(tf.expand_dims(y_len,axis=1),5,axis=1)
    #print('gt_idx shape: {}'.format(gt_idx.shape))
    #print('pred_idx shape: {}'.format(pred_idx.shape))
    #print('filled shape: {}'.format(filled_length.shape))

    # b, 5, 3
    stacked = tf.stack([gt_idx, pred_idx, filled_length], axis=-1)

    # sum the error per entity, and calc the mean over the batches
    # for each batch ==> 5, 3 in stacked
    diffs = tf.map_fn(lambda x: get_min_dist_for_list(x), stacked, dtype=tf.int32)
    if apply_sum: diffs = tf.cast(tf.reduce_sum(diffs, axis=1),tf.float32)
    if apply_average: diffs = tf.reduce_mean(diffs)
    tf.math.greater_equal(diffs, 0.), 'distance cant be smaller than 0'
    return diffs

@tf.function
def get_min_dist_for_list(vals):
    # vals has the shape 5, 3
    # for each phase tuple (gt,pred,length)
    return tf.map_fn(lambda x :get_min_distance(x),vals, dtype=tf.int32)
@tf.function
def get_min_distance(vals):

    smaller = tf.reduce_min(vals[0:2], keepdims=True)
    bigger = tf.reduce_max(vals[0:2], keepdims=True)
    mod = vals[2]

    diff = bigger - smaller
    diff_ring = tf.math.abs(mod - bigger + smaller)# we need to use the abs to avoid 0 - 0
    min_diff = tf.reduce_min(tf.stack([diff, diff_ring]))
    tf.math.greater_equal(min_diff, 0) # this is an int, as we measure the distance between buckets for the metric
    return min_diff

class Meandiff_loss():

    def __init__(self):
        self.__name__ = 'meandiff_loss'

    def __call__(self, y_true, y_pred, **kwargs):
        return  meandiff_loss(y_true, y_pred, apply_sum=True, apply_average=True) # this should yield a loss between 1 and 0.0001


def meandiff_loss( y_true, y_pred, apply_sum=True, apply_average=True, as_loss=False):

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
    # split gt mask and onehot
    y_true, y_len_msk = tf.unstack(y_true,2,axis=1)
    y_pred, _ = tf.unstack(y_pred,2,axis=1)

    # convert to tensor
    y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
    y_len_msk = tf.cast(tf.convert_to_tensor(y_len_msk), tf.float32)

    # multiply with mask,
    # we are interested in the time step per phase within the gt length
    # b, 36, 5
    temp_pred = y_pred * y_len_msk
    temp_gt = y_true * y_len_msk

    # make sure axis 1 sums up to one
    #temp_pred = tf.keras.activations.softmax(temp_pred, axis=1)
    #temp_gt = tf.keras.activations.softmax(temp_gt, axis=1)

    #temp_pred = y_pred
    #temp_gt = y_true

    # get the original lengths of each mask in the current batch
    # b, 1
    y_len = tf.cast(tf.reduce_sum(y_len_msk[:,:,0], axis=1),dtype=tf.float32)#

    ############################################ naive test, this works in eager, but not in the loss (line: tf.tile...)
    """@tf.function
    def helper_max(temp):
        # b,36, 5
        max_ = tf.reduce_max(temp, axis=1, keepdims=True)
        # the max value per phase
        # b,1,5, we are interested in b,5 each value in "5" points to the timestep where this phase occures
        # get a mask which points to the max value along axis 1 (for all 36 timesteps)
        cond = tf.cast((temp == max_), tf.float32)
        # b,36,5
        pos = tf.cast(tf.range(36), tf.float32)
        pos = tf.expand_dims(tf.expand_dims(pos, axis=0), axis=-1)
        pos = tf.tile(pos, (temp.shape[0], 1, temp.shape[-1]))
        soft_pos = tf.reduce_sum(cond * pos, axis=1)
        return soft_pos
    gt_idx = helper_max(temp_gt)
    pred_idx = helper_max(temp_pred)"""
    ################################################

    gt_idx = tf.cast(DifferentiableArgmax(temp_gt, axis=1), dtype=tf.float32)#
    pred_idx = tf.cast(DifferentiableArgmax(temp_pred, axis=1), dtype=tf.float32)
    filled_length = tf.repeat(tf.expand_dims(y_len,axis=1),5,axis=1)
    #print('gt_idx shape: {}'.format(gt_idx.shape))
    #print('pred_idx shape: {}'.format(pred_idx.shape))
    #print('filled shape: {}'.format(filled_length.shape))

    # b, 5, 3
    stacked = tf.stack([gt_idx, pred_idx, filled_length], axis=-1)

    # sum the error per entity, and calc the mean over the batches
    # for each batch ==> 5, 3 in stacked
    diffs = tf.map_fn(lambda x: get_min_dist_for_list_loss(x), stacked, dtype=tf.float32)
    if apply_sum: diffs = tf.cast(tf.reduce_sum(diffs, axis=1),tf.float32)
    if apply_average: diffs = tf.reduce_mean(diffs)
    #tf.math.greater_equal(diffs, 0.), 'distance cant be smaller than 0'
    return diffs

@tf.function
def get_min_dist_for_list_loss(vals):
    # vals has the shape 5, 3
    # for each phase tuple (gt,pred,length)
    return tf.map_fn(lambda x :get_min_distance_loss(x),vals, dtype=tf.float32)
@tf.function
def get_min_distance_loss(vals):

    smaller = tf.reduce_min(vals[0:2], keepdims=True)
    bigger = tf.reduce_max(vals[0:2], keepdims=True)
    mod = vals[2]

    diff = bigger - smaller
    diff_ring = tf.math.abs(mod - bigger + smaller)# we need to use the abs to avoid 0 - 0
    min_diff = tf.reduce_min(tf.stack([diff, diff_ring]))
    #tf.math.greater_equal(min_diff, 0.)
    return min_diff

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Lambda


@tf.function
def DifferentiableArgmax(inputs, axis=-1):
    # if it doesnt sum to one: normalize
    @tf.function
    def prob2oneHot(x):
        # len should be slightly larger than the length of x
        len = 40
        a = tf.math.pow(len * x, 10)
        sum_a = tf.reduce_sum(a, axis=axis)
        sum_a = tf.expand_dims(sum_a, axis=axis)
        onehot = tf.divide(a, sum_a)

        return onehot

    onehot = prob2oneHot(inputs)
    onehot = prob2oneHot(onehot)
    onehot = prob2oneHot(onehot)

    @tf.function
    def onehot2token(x):
        cumsum = tf.cumsum(x, axis=axis, exclusive=True, reverse=True)
        rounding = 2 * (tf.clip_by_value(cumsum, clip_value_min=.5, clip_value_max=1) - .5)
        token = tf.reduce_sum(rounding, axis=axis)
        return token

    token = onehot2token(onehot)
    return token

class CCE_combined(tf.keras.losses.Loss):

    def __init__(self, masked=False, smooth=0, transposed=False):

        super().__init__(name='cce_combined')
        self.masked = masked
        self.smooth = smooth
        self.transposed = transposed
        self.cce = CCE(masked=False,smooth=smooth,transposed=False)
        self.cce_t = CCE(masked=False, smooth=smooth, transposed=True)
        self.cce_weight = 0.5
        self.cce_t_weight = 0.5

    def __call__(self, y_true, y_pred, **kwargs):

        return self.cce_weight*self.cce(y_true, y_pred) + self.cce_t_weight * self.cce_t(y_true,y_pred)


class CCE(tf.keras.losses.Loss):

    def __init__(self, masked=False, smooth=0, transposed=False):

        super().__init__(name='cce_{}_{}_{}'.format(masked,smooth,transposed))
        self.masked = masked
        self.smooth = smooth
        self.transposed = transposed

    def __call__(self, y_pred, y_true, **kwargs):

        if y_true.shape[1] == 2: # this is a stacked onehot vector
            y_true, y_msk = tf.unstack(y_true, num=2, axis=1)
            y_pred, _ = tf.unstack(y_pred, num=2, axis=1)

            if self.masked:
                y_true = y_true * y_msk
                y_pred = y_pred * y_msk

        if self.transposed:
            y_true = tf.nn.softmax(tf.transpose(y_true, perm=[0, 2, 1]), axis=-1)
            y_pred = tf.nn.softmax(tf.transpose(y_pred, perm=[0, 2, 1]), axis=-1)

        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=self.smooth)

        return loss




def cce_wrapper_transpose(y_true,y_pred, masked=False):
    y_true, y_msk = tf.unstack(y_true, num=2, axis=1)
    y_pred, _ = tf.unstack(y_pred, num=2, axis=1)

    if masked: y_true = y_true * y_msk
    if masked: y_pred = y_pred * y_msk

    y_true = tf.nn.softmax(tf.transpose(y_true, perm=[0, 2, 1]), axis=-1)
    y_true = tf.nn.softmax(tf.transpose(y_true, perm=[0, 2, 1]), axis=-1)

    loss = tf.keras.losses.categorical_crossentropy(y_true,y_pred, label_smoothing=0.5)
    loss = tf.transpose(loss, perm=[0,2,1])

    return loss


class MSE(tf.keras.losses.Loss):

    def __init__(self, masked=False):

        super().__init__(name='MSE_{}'.format(masked))
        self.masked = masked


    def __call__(self, y_pred, y_true, **kwargs):

        if y_true.shape[1] == 2: # this is a stacked onehot vector
            y_true, y_msk = tf.unstack(y_true, num=2, axis=1)
            y_pred, _ = tf.unstack(y_pred, num=2, axis=1)


            if self.masked:
                y_msk = tf.cast(y_pred, tf.float32) + 1  # weight the first area by 2
                y_true = y_msk * y_true
                y_pred =  y_msk * y_pred

        return tf.keras.losses.mse(y_true, y_pred)

def mse_wrapper(y_true,y_pred):
    y_true, y_len = tf.unstack(y_true,num=2, axis=1)
    y_pred, _ = tf.unstack(y_pred,num=2, axis=1)
    return tf.keras.losses.mse(y_true, y_pred)

def ca_wrapper(y_true, y_pred):
    y_true, y_len = tf.unstack(y_true,num=2, axis=1)
    y_pred, _ = tf.unstack(y_pred,num=2, axis=1)
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)
