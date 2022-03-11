import logging
from src.models.KerasLayers import ConvEncoder, get_angle_tf, get_idxs_tf, get_centers_tf, ComposeTransform, \
    conv_layer_fn, ConvBlock

import sys
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow import keras
import keras.layers as KL
from keras.layers import Input
from keras.models import Model
from tensorflow.python.keras import metrics as metr
from keras.layers import Dropout, BatchNormalization, TimeDistributed
from keras.layers import LSTM, Bidirectional
import math
import atexit

from src.models.Unets import create_unet
from src.utils import Metrics as own_metr

from src.models.ModelUtils import get_optimizer

sys.path.append('src/ext/neuron')
sys.path.append('src/ext/pynd-lib')
sys.path.append('src/ext/pytools-lib')
import src.ext.neuron.neuron.layers as nrn_layers

def create_PhaseRegressionModel_v2(config, networkname='PhaseRegressionModel'):
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the "mirrored data"-paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    with strategy.scope():

        """from tensorflow.keras import mixed_precision
        policy = mixed_precision.experimental.Policy('mixed_float16')
        mixed_precision.experimental.set_policy(policy)"""

        input_shape = config.get('DIM', [10, 224, 224])
        T_SHAPE = config.get('T_SHAPE', 40)
        input_tensor = Input(shape=(T_SHAPE, *input_shape, 1))
        # define standard values according to the convention over configuration paradigm
        dim = config.get('DIM', [10, 224, 224])
        ndims = len(config.get('DIM', [10, 224, 224]))
        add_vect_norm = config.get('ADD_VECTOR_NORM', False)
        add_vect_direction = config.get('ADD_VECTOR_DIRECTION', False)
        add_flows = config.get('ADD_FLOW', False)
        image_loss_weight = config.get('IMAGE_LOSS_WEIGHT', 20)
        feature_loss_weight = config.get('FEATURE_LOSS_WEIGHT', 1)
        flow_loss_weight = config.get('FLOW_LOSS_WEIGHT', 0.01)
        loss = config.get('LOSS', 'mse').lower()
        mask_loss = config.get('MASK_LOSS', False)
        interp_method = 'linear'
        indexing = 'ij'
        # TODO: this parameter is also used by the generator to define the number of channels
        # here we stack the volume within the model manually, so keep that aligned!!!
        temp_config = config.copy()  # dont change the original config
        temp_config['IMG_CHANNELS'] = 2  # we roll the temporal axis and stack t-1, t and t+1 along the last axis
        stack_axis = 1

        ############################# definition of the layers and blocks ######################################
        # start with very small deformation
        Conv = getattr(KL, 'Conv{}D'.format(ndims))
        Conv_layer = Conv(ndims, kernel_size=3, padding='same',
                          kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5),
                          name='unet2flow')
        # use a standard U-net, without the final layer for feature extraction (pre-flow)
        unet = create_unet(temp_config, single_model=False, networkname='3D-Unet')
        # this is a wrapper to re-use the u-net encoder/encoding
        enc = keras.Model(inputs=unet.inputs, outputs=[unet.layers[(len(unet.layers) // 2) - 1].output])
        st_layer = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True,
                                                 name='deformable_layer')
        st_lambda_layer = keras.layers.Lambda(
            lambda x: st_layer([x[..., 0:1], x[..., 1:]]), name='deformable_lambda_layer')

        gap = tensorflow.keras.layers.GlobalAveragePooling3D(name='GAP_3D_Layer')
        # concat the current frame with the previous on the last channel, this is not valid for the last frame

        roll_concat_lambda_layer = keras.layers.Lambda(lambda x:
                                                       keras.layers.Concatenate(axis=-1, name='stack_with_moved')(
                                                           [  # tf.roll(x, shift=1, axis=stack_axis),
                                                               x,
                                                               tf.roll(x, shift=-1, axis=stack_axis)]))

        norm_lambda = keras.layers.Lambda(
            lambda x: tf.norm(x, ord='euclidean', axis=-1, keepdims=True, name='flow2norm'), name='flow2norm')

        # calculate the direction between the displacement field and a grid with vectors pointing to the center
        # get a tensor with vectors pointing to the center
        # get idxs of one 3D
        # get a tensor with the same shape as the displacement field with vectors toward the center
        # calculate the difference, which should yield a 3D tensor with vectors pointing to the center
        # tile/repeat this v_center vol along the temporal and batch axis
        # calculate the angle of each voxel between the tiled v_center tensor and the displacement tensor
        # concat this tensor as additional feature to the last axis of flow_features
        idx = get_idxs_tf(dim)
        c = get_centers_tf(dim)
        # print('centers: ',c.dtype)
        centers = c - idx
        centers_tensor = centers[tf.newaxis, ...]
        flow2direction_lambda = keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')

        stack_lambda_tf = keras.layers.Lambda(lambda x:
                                              tf.stack([x,
                                                        tf.zeros_like(x, name='zero_padding')],
                                                       axis=1, name='extend_onehot_by_zeros'),
                                              name='onehot_lambda')



        ##################################### Layer definition end ##############################################

        # unstack along t yielding a list of 3D volumes
        # roll t-1, t and t+1 (axis 1) and stack on axis -1, yielding a temporal window for the 3D conv net

        print('Shape Input Tensor: {}'.format(input_tensor.shape))
        inputs_spatial_stacked = roll_concat_lambda_layer(input_tensor)
        # replace the last timestep my the 2nd last timestep, otherwise we might try to predict
        # the motion from the middle of a cardiac cycle to the first timestep (ED)
        # repeat the last time step
        # For the last timestep
        # this will result in different model input t,t and loss target t,t+1
        # inputs_spatial_stacked = keras.layers.Concatenate(axis=1)([inputs_spatial_stacked[:,:-1], inputs_spatial_stacked[:,-2:-1]])
        print('Shape rolled and stacked: {}'.format(inputs_spatial_stacked.shape))
        pre_flows = TimeDistributed(unet, name='4d-p2p-unet')(inputs_spatial_stacked)
        print('Unet output shape: {}'.format(pre_flows.shape))
        flows = TimeDistributed(Conv_layer, name='4d-p2p-flow')(pre_flows)
        print('Flowfield shape: {}'.format(flows.shape))
        transformed = TimeDistributed(st_lambda_layer, name='4d-p2p-st')(
            keras.layers.Concatenate(axis=-1)([input_tensor, flows]))
        print('Transformed shape : {}'.format(transformed.shape))
        features_given = False

        if (add_vect_norm and add_flows):
            # add the magnitude as fourth channel
            tensor_magnitude = TimeDistributed(norm_lambda)(flows)
            flow_features = keras.layers.Concatenate(axis=-1)([flows, tensor_magnitude])
            features_given = True
            print('Inkl flow and norm shape: {}'.format(flow_features.shape))
        elif add_vect_norm:
            # add the magnitude as fourth channel
            tensor_magnitude = TimeDistributed(norm_lambda)(flows)
            flow_features = tensor_magnitude
            features_given = True
            print('Inkl norm shape: {}'.format(flow_features.shape))
        elif add_flows:
            flow_features = flows
            features_given = True
            print('Inkl flow shape: {}'.format(flow_features.shape))

        if add_vect_direction:
            directions = TimeDistributed(flow2direction_lambda)(flows)
            print('directions shape: {}'.format(directions.shape))
            if features_given:
                flow_features = keras.layers.Concatenate(axis=-1)(
                    [flow_features, directions])  # encode the spatial location of each vector
            else:
                flow_features = directions
                features_given = True
            # add the location tensor as further channel
            # flow_features = keras.layers.Concatenate(axis=-1)([flow_features, tf.tile(centers_tensor[tf.newaxis,...],multiples=[1,flow_features.shape[1],1,1,1,1])])
        print('flow features inkl directions shape: {}'.format(flow_features.shape))


        # define the model output names
        flow_features = keras.layers.Activation('linear', dtype='float32', name='features')(flow_features)
        transformed = keras.layers.Activation('linear', name='transformed', dtype='float32')(transformed)
        flows = keras.layers.Activation('linear', name='flows', dtype='float32')(flows)

        outputs = [flow_features, transformed, flows]
        from keras.losses import mse
        from src.utils.Metrics import Grad

        weights = {
            'features': feature_loss_weight,
            'transformed': image_loss_weight,
            'flows': flow_loss_weight}

        if loss == 'ssim':
            losses = {
                #'features': None,
                'transformed': own_metr.SSIM(),
                'flows': Grad('l2').loss}
        elif loss == 'mae':
            losses = {
                #'features': None,
                'transformed': own_metr.MSE(masked=mask_loss, loss_fn=keras.losses.mae, onehot=False),
                'flows': Grad('l2').loss}

        else:  # default fallback --> MSE - works the best
            losses = {
                #'features': own_metr.MSE(masked=mask_loss, loss_fn=keras.losses.mse, onehot=True),
                'transformed': own_metr.MSE(masked=mask_loss, loss_fn=keras.losses.mse, onehot=False),
                'flows': Grad('l2').loss}

        print('added loss: {}'.format(loss))
        model = Model(inputs=[input_tensor], outputs=outputs, name=networkname)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.get('LEARNING_RATE', 0.001)),
            loss=losses,
            loss_weights=weights,
            metrics={
                'transformed': own_metr.MSE(masked=mask_loss, loss_fn=keras.losses.mse, onehot=False),
            }
        )
        """[print(i.shape, i.dtype) for i in model.inputs]
        [print(o.shape, o.dtype) for o in model.outputs]
        [print(l.name, l.input_shape, l.dtype) for l in model.layers]"""
        return model



# ST to apply m to an volume
def create_affine_transformer_fixed(config, networkname='affine_transformer_fixed', fill_value=0,
                                    interp_method='linear'):
    """
    Apply a learned transformation matrix to an input image, no training possible
    :param config:  Key value pairs for image size and other network parameters
    :param networkname: string, name of this model scope
    :param fill_value:
    :return: compiled keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))

    with strategy.scope():

        inputs = Input((*config.get('DIM', [10, 224, 224]), 1))
        input_displacement = Input((*config.get('DIM', [10, 224, 224]), 3))
        indexing = config.get('INDEXING', 'ij')

        # warp the source with the flow
        y = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=False,
                                          fill_value=fill_value)([inputs, input_displacement])

        model = Model(inputs=[inputs, input_displacement], outputs=[y, input_displacement], name=networkname)

        return model


def create_dense_compose(config, networkname='dense_compose_displacement'):
    """
    Compose a single transform from a series of transforms.
    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:
    T = compose([A, B, C])
    :param config:  Key value pairs for image size and other network parameters
    :param networkname: string, name of this model scope
    :param fill_value:
    :return: compiled keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))

    with strategy.scope():

        inputs = Input((5, *config.get('DIM', [10, 224, 224]), 3))
        indexing = config.get('INDEXING', 'ij')
        reverse = config.get('REVERSE_COMPOSE', False)
        # warp the source with the flow
        flows = tf.unstack(inputs, axis=1)
        # reverse=True
        # we need to reverse the transforms as we register from t+1 to t.
        # we need to provide the order in which we would apply the compose transforms
        # ED, MS, ES, PF, MD
        # flows:
        # 0= MS->ED, 1=ES->MS, 2=PF->ES, 3=MD->PF, 4=ED->MD
        # e.g. for compose:
        # MS->ED = [0]
        # ES->ED = [1,0]
        # PF->ED = [2,1,0]
        # list(reversed())
        if reverse:
            y = [ComposeTransform(interp_method='linear', shift_center=True, indexing=indexing,
                                  name='Compose_transform{}'.format(i))(list(reversed(flows[:i]))) for i in
                 range(2, len(flows) + 1)]
        else:
            y = [ComposeTransform(interp_method='linear', shift_center=True, indexing=indexing,
                                  name='Compose_transform{}'.format(i))(flows[:i]) for i in range(2, len(flows) + 1)]
        y = tf.stack([flows[0], *y], axis=1)

        model = Model(inputs=[inputs], outputs=[y], name=networkname)

        return model
