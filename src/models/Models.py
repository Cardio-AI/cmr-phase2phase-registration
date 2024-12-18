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
from src.utils.Metrics import MSE

sys.path.append('src/ext/neuron')
sys.path.append('src/ext/pynd-lib')
sys.path.append('src/ext/pytools-lib')
import src.ext.neuron.neuron.layers as nrn_layers


def create_PhaseRegressionModel(config, networkname='PhaseRegressionModel'):
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
        atexit.register(strategy._extended._collective_ops._pool.close)
    else:
        # distribute the training with the "mirrored data"-paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        input_shape = config.get('DIM', [10, 224, 224])
        T_SHAPE = config.get('T_SHAPE', 36)
        PHASES = config.get('PHASES', 5)
        input_tensor = Input(shape=(T_SHAPE, *input_shape, 1))
        # define standard values according to the convention over configuration paradigm
        activation = config.get('ACTIVATION', 'elu')
        batch_norm = config.get('BATCH_NORMALISATION', False)
        pad = config.get('PAD', 'same')
        kernel_init = config.get('KERNEL_INIT', 'he_normal')
        m_pool = config.get('M_POOL', (1, 2, 2))
        f_size = config.get('F_SIZE', (3, 3, 3))
        filters = config.get('FILTERS', 16)
        drop_1 = config.get('DROPOUT_MIN', 0.3)
        drop_3 = config.get('DROPOUT_MAX', 0.5)
        bn_first = config.get('BN_FIRST', False)
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)
        batchsize = config.get('BATCHSIZE', 8)
        add_bilstm = config.get('ADD_BILSTM', False)
        lstm_units = config.get('BILSTM_UNITS', 64)
        final_activation = config.get('FINAL_ACTIVATION', 'relu').lower()
        loss = config.get('LOSS', 'mse').lower()
        mask_loss = config.get('MASK_LOSS', False)
        pre_gap_conv = config.get('PRE_GAP_CONV', False)

        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]

        temporal_encoder = ConvEncoder(activation=activation,
                                       batch_norm=batch_norm,
                                       bn_first=bn_first,
                                       depth=depth,
                                       drop_3=drop_3,
                                       dropouts=dropouts,
                                       f_size=f_size,
                                       filters=filters,
                                       kernel_init=kernel_init,
                                       m_pool=m_pool,
                                       ndims=ndims,
                                       pad=pad)

        gap = tensorflow.keras.layers.GlobalAveragePooling3D()

        # unstack along the temporal axis
        # added shuffling, which avoids the model to be biased by the order
        # unstack along t, yielding a list of 3D volumes

        import random
        # unstack along Z yielding a list of 2D+t slices
        inputs_temporal = tf.unstack(input_tensor, axis=2)
        indicies = list(tf.range(len(inputs_temporal)))
        zipped = list(zip(inputs_temporal, indicies))
        random.shuffle(zipped)
        inputs_temporal, indicies = zip(*zipped)
        # feed the 2D+T slices as 3D volumes into the temporal encoder (3D conv)
        inputs_temporal = [temporal_encoder(vol)[0] for vol in inputs_temporal]
        inputs_temporal, _ = zip(*sorted(zip(inputs_temporal, indicies), key=lambda tup: tup[1]))
        # stack the 2D+T encoding along the spatial axis (2, as we have b,t,z,x,y,c) --> 3D+T encoding
        inputs_temporal = tf.stack(inputs_temporal, axis=2)
        print('Shape after the temporal encoder')
        print(inputs_temporal.shape)
        # unstack each 3D volume encoding, apply global average pooling on each
        inputs_temporal = tf.unstack(inputs_temporal, axis=1)

        if pre_gap_conv:
            gap_conv = keras.layers.Conv3D(filters=64, kernel_size=3, strides=(2, 1, 1), padding='valid',
                                              activation=activation, kernel_initializer=kernel_init)

            inputs_temporal = [keras.layers.BatchNormalization()(elem) for elem in inputs_temporal]
            inputs_temporal = [keras.layers.Dropout(rate=0.5)(elem) for elem in inputs_temporal]
            inputs_temporal = [gap_conv(vol) for vol in inputs_temporal]
        inputs_temporal = [gap(vol) for vol in inputs_temporal]
        inputs_temporal = tf.stack(inputs_temporal, axis=1)

        inputs = inputs_temporal
        print('Shape after GAP')
        print(inputs.shape)
        # 36, 256
        if add_bilstm:
            print('add a bilstm layer with: {} lstm units'.format(lstm_units))
            """inputs = keras.layers.BatchNormalization()(inputs)
            inputs = keras.layers.Dropout(rate=0.5)(inputs)
    
            onehot_pre = keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation=activation,
                                            name='conv_pre_final')(inputs)"""
            forward_layer = LSTM(lstm_units, return_sequences=True)
            backward_layer = LSTM(lstm_units, activation='tanh', return_sequences=True,
                                  go_backwards=True)  # maybe change to tanh
            inputs = Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=inputs.shape)(inputs)

        # 36,64
        print('Shape after Bi-LSTM layer')
        print(inputs.shape)

        # input (36,encoding) output (36,5)
        # either 36,256 --> from the temp encoder or
        # 36,64 --> 64 --> number of BI-LSTM units
        # activation to linear
        onehot = keras.layers.Conv1D(filters=PHASES, kernel_size=1, strides=1, padding='same', activation='linear',
                                        name='final_conv')(inputs)
        if final_activation == 'relu':
            onehot = keras.activations.relu(onehot)
        elif final_activation == 'softmax':
            # axis -1 --> one class per timestep, as we repeat the phases its not possible to softmax the phase
            onehot = keras.activations.softmax(onehot, axis=-1)
        elif final_activation == 'sigmoid':
            onehot = keras.activations.sigmoid(onehot)
        else:
            logging.info('No final activation given! Please check the "FINAL_ACTIVATION" param!')

        # 36, 5
        print('Shape after final conv layer')
        print(onehot.shape)

        # add empty tensor with one-hot shape to align with gt
        zeros = tf.zeros_like(onehot)
        onehot = tf.stack([onehot, zeros], axis=1)

        outputs = [onehot]

        if loss == 'cce':
            losses = [own_metr.CCE(masked=mask_loss, smooth=0.8, transposed=False)]
        elif loss == 'meandiff':
            losses = [own_metr.Meandiff_loss()]
        elif loss == 'msecce':
            print(loss)
            # own_metr.CCE(masked=False,smooth=0.8,transposed=False)
            losses = [own_metr.MSE(), own_metr.CCE(masked=mask_loss, smooth=0.8, transposed=False)]
            print(losses)
        else:  # default fallback --> MSE - works the best
            losses = [own_metr.MSE(masked=mask_loss)]

        print('added loss: {}'.format(loss))
        model = Model(inputs=[input_tensor], outputs=outputs, name=networkname)
        model.compile(
            optimizer=get_optimizer(config, networkname),
            loss=losses,
            # metrics=[own_metr.mse_wrapper, own_metr.ca_wrapper, own_metr.meandiff] #
            metrics=[own_metr.meandiff]  #
        )

        return model


def create_RegistrationModel(config):
    """
    A registration wrapper for 3D image2image registration
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the "mirrored data"-paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    with strategy.scope():
        if config is None:
            config = {}
        input_shape = config.get('DIM', [10, 224, 224])
        T_SHAPE = config.get('T_SHAPE', 5)
        image_loss_weight = config.get('IMAGE_LOSS_WEIGHT', 1)
        reg_loss_weight = config.get('REG_LOSS_WEIGHT', 0.001)
        learning_rate = config.get('LEARNING_RATE', 0.001)

        input_tensor = Input(shape=(T_SHAPE, *input_shape, config.get('IMG_CHANNELS',
                                                                      1)))  # input vol with timesteps, z, x, y, c -> =number of input timesteps
        input_tensor_empty = Input(shape=(T_SHAPE, *input_shape, 3))  # empty vector field
        # define standard values according to the convention over configuration paradigm

        ndims = len(input_shape)
        indexing = 'ij'
        interp_method = 'linear'
        Conv = getattr(KL, 'Conv{}D'.format(ndims))
        take_t_elem = config.get('INPUT_T_ELEM', 0)

        # start with very small deformation
        Conv_layer = Conv(ndims, kernel_size=3, padding='same',
                          kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5),
                          name='unet2flow')
        st_layer = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True,
                                                 name='deformable_layer')

        unet = create_unet(config, single_model=False)

        input_vols = tf.unstack(input_tensor, axis=1)
        print(input_vols[0].shape)
        import random
        indicies = list(tf.range(len(input_vols)))
        zipped = list(zip(input_vols, indicies))
        random.shuffle(zipped)
        input_vols_shuffled, indicies = zip(*zipped)
        # input_vols_shuffled = input_vols
        pre_flows = [unet(vol) for vol in input_vols_shuffled]
        flows = [Conv_layer(vol) for vol in pre_flows]
        flows, _ = zip(*sorted(zip(flows, indicies), key=lambda tup: tup[1]))

        transformed = [st_layer([input_vol[..., take_t_elem][..., tf.newaxis], flow]) for input_vol, flow in
                       zip(input_vols, flows)]
        transformed = tf.stack(transformed, axis=1)
        flow = tf.stack(flows, axis=1)

        outputs = [transformed, flow]

        model = Model(name='simpleregister', inputs=[input_tensor, input_tensor_empty], outputs=outputs)

        from keras.losses import mse
        from src.utils.Metrics import Grad, MSE_

        losses = [MSE_().loss, Grad('l2').loss]
        weights = [image_loss_weight, reg_loss_weight]
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=losses,
                      loss_weights=weights)

    return model


def create_RegistrationModel_inkl_mask(config):
    """
    A registration wrapper for 3D image2image registration
    """
    import random
    from keras.losses import mse
    from src.utils.Metrics import Grad, MSE_, SSIM, NormRegulariser
    from src.utils.Metrics import dice_coef_loss
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the "mirrored data"-paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
    with strategy.scope():
        if config is None:
            config = {}
        input_shape = config.get('DIM', [10, 224, 224])
        T_SHAPE = config.get('T_SHAPE', 5)
        image_loss_weight = config.get('IMAGE_LOSS_WEIGHT', 1)
        dice_loss_weight = config.get('MASK_LOSS_WEIGHT', 0.1)
        reg_loss_weight = config.get('REG_LOSS_WEIGHT', 0.001)
        learning_rate = config.get('LEARNING_RATE', 0.001)
        COMPOSE_CONSISTENCY = config.get('COMPOSE_CONSISTENCY', False)
        register_backwards = config.get('REGISTER_BACKWARDS', True)
        register_spatial = config.get('REGISTER_SPATIAL', False)
        image_loss = config.get('IMAGE_LOSS', 'mse').lower()
        image_comp_loss = config.get('IMAGE_COMP_LOSS', 'mse').lower()
        img_flow_reg_loss = config.get('IMG_REG_LOSS', 'grad').lower()
        img_comp_flow_reg_loss = config.get('IMG_COMP_REG_LOSS', 'grad').lower()
        dedicated_unet = config.get('DEDICATED_UNET', True)

        if image_loss == 'ssim':
            image_loss_fn = SSIM()
            print('ssim')
        else:
            image_loss_fn = MSE()

        if image_comp_loss =='ssim':
            image_comp_loss_fn = SSIM()
            print('ssim')
        else:
            image_comp_loss_fn = MSE()

        if img_flow_reg_loss == 'norm':
            flow_reg_loss_fn = NormRegulariser().norm_loss
        else:
            flow_reg_loss_fn = Grad('l2',loss_mult=tuple(config.get('SPACING',(1,1,1)))).loss

        if img_comp_flow_reg_loss == 'norm':
            flow_comp_reg_loss_fn = NormRegulariser().norm_loss
        else:
            flow_comp_reg_loss_fn = Grad('l2',loss_mult=tuple(config.get('SPACING',(1,1,1)))).loss


        config_temp = config.copy()

        # input vol with timesteps, z, x, y, c -> =number of input timesteps
        stacked_cmr = 3
        input_tensor_raw = Input(shape=(T_SHAPE, *input_shape, stacked_cmr), name='cmr')
        input_mask_tensor = Input(shape=(T_SHAPE, *input_shape, config.get('IMG_CHANNELS', 1)), name='mask')
        # define standard values according to the convention over configuration paradigm

        ndims = len(input_shape)
        indexing = 'ij'
        interp_method = 'linear'
        Conv = getattr(KL, 'Conv{}D'.format(ndims))
        take_t_elem = config.get('INPUT_T_ELEM', 0)

        # start with very small deformation
        conv_layer_p2p = Conv(3, kernel_size=3, padding='same',
                              kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=1e-10),
                              name='unet2flow_p2p')
        conv_layer_p2ed = Conv(3, kernel_size=3, padding='same',
                               kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=1e-10),
                               name='unet2flow_p2ed')
        st_layer_p2p = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True,
                                                 name='deformable_p2p')
        st_layer_p2ed = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True,
                                                      name='deformable_p2ed')
        st_mask_p2p_layer = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True,
                                                      name='deformable_mask')
        st_mask_p2ed_layer = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True,
                                                      name='deformable_mask')

        # combine moving image and deformable
        # if we call the st layer directly, we would have slicing layers in the model plot
        st_lambda_layer = keras.layers.Lambda(
            lambda x: st_layer_p2p([x[..., :1], x[..., -3:]]), name='p2p')
        st_p2ed_lambda_layer = keras.layers.Lambda(
            lambda x: st_layer_p2ed([x[..., :1], x[..., -3:]]), name='p2ed')
        st_mask_p2p_lambda_layer = keras.layers.Lambda(
            lambda x: st_mask_p2p_layer([x[..., :1], x[..., -3:]]), name='p2p_mask')
        st_mask_p2ed_lambda_layer = keras.layers.Lambda(
            lambda x: st_mask_p2ed_layer([x[..., :1], x[..., -3:]]), name='p2ed_mask')

        # lambda layers for spatial transformer indexing of the cmr vol and the deformable
        # deformable follows ij indexing --> z,y,x
        if register_spatial: # keep the flow as it is
            add_zero_spatial_lambda_p2p = keras.layers.Lambda(
                lambda x: x, name='keep_spatial_p2p')
            add_zero_spatial_lambda_p2ed = keras.layers.Lambda(
                lambda x: x, name='keep_spatial_p2ed')
            print('register in 3D')
        else: # ignore the z-axis for the spatial transformer, only x/y movement
            # zero out the z motion
            # concat zero with the  y and x deformable

            add_zero_spatial_lambda_p2p = keras.layers.Lambda(
                lambda x: tf.concat([tf.zeros_like(x[..., -1:]), x[..., -2:]], axis=-1), name='del_spatial_p2p')
            add_zero_spatial_lambda_p2ed = keras.layers.Lambda(
                lambda x: tf.concat([tf.zeros_like(x[..., -1:]), x[..., -2:]], axis=-1), name='del_spatial_p2ed')


            # extract the ed phase as volume
            # repeat ED along the t-axis
            # add the ed phase as 4th channel to each phase
            # replace t shifted by the ed phase as second input for the compose flow
            # x[:, 0:1, ..., -1:] --> the shape is: Batchsize, Phases, Z, X, Y, C
            # The order of the volumes in our channel depends on the parameter "register_backwards",
            # if register_backwards: Channel0 == Phase-1 (shift to the left); Channel1 == Phase
            # else: Channel0 == Phase; Channel1 == Phase-1 (shift to the left)
            # Here we slice the ED 3D volume and choose the last channel, which represents the actual frame
            # in our stack lambda layers we use the first Channel for transformation
        if register_backwards: # here the ED should be our target
            # x = cmr = x_k #ED, MS, ES, PF, MD
            # x2 = mask = x_k #ED, MS, ES, PF, MD
            # y = ED-w, MS-w, ES-w, PF-w, MD-w
            # Y_ed = ED, ED, ED, ED, ED

            stack_p2p_lambda_layer = keras.layers.Lambda(lambda x: x[...,:2],name='stack_p2p')

            stack_ed_lambda_layer = keras.layers.Lambda(
                lambda x: keras.layers.Concatenate(axis=-1)(
                    [x [...,:1], # ED, MS, ES, PF, MD
                     x[...,-1:], # ED, ED, ED, ED, ED

                     ]),
                name='stack_ed')



        else: # register forwards, x is in this case x_t, e.g.: x_0 = ED
            raise NotImplementedError('need to check this flow')
            stack_p2p_lambda_layer = keras.layers.Lambda(
                lambda x: keras.layers.Concatenate(axis=-1)(
                    [x, # ED,MS,ES,PF,MD
                     tf.roll(x, shift=-1, axis=1), # MS,ES,PF,MD,ED
                     #tf.math.squared_difference(x,tf.roll(x, shift=-1, axis=1))
                     ]),
                name='stack_p2p')

            stack_ed_lambda_layer = keras.layers.Lambda(
                lambda x: keras.layers.Concatenate(axis=-1)(
                    [tf.repeat(x[:, 0:1, ...], repeats=5, axis=1), # ED, ED, ED, ED, ED
                    tf.roll(x, shift=-1, axis=1), # MS,ES,PF,MD,ED
                     #tf.math.squared_difference(tf.repeat(x[:, 0:1, ...], repeats=5, axis=1),tf.roll(x, shift=-1, axis=1))
                     ]),
                name='stack_ed')

        input_tensor = stack_p2p_lambda_layer(input_tensor_raw)

        # we need to build the u-net after the compose concat path to make sure that our u-net input channels match the input
        config_temp['IMG_CHANNELS'] = input_tensor.shape[-1]
        print('p2p input channels:', input_tensor.shape[-1])
        unet = create_unet(config_temp, single_model=False)
        print('input before unet:', input_tensor.shape)
        pre_flows = TimeDistributed(unet, name='unet')(input_tensor)
        print('input after unet:', pre_flows.shape)
        flows = TimeDistributed(conv_layer_p2p, name='unet2flow_p2p')(pre_flows)
        flows = add_zero_spatial_lambda_p2p(flows)
         # according to the Tensorboard order of the Channel C --> is zyx.
        print('flows_p2p:', flows.shape)
        # Each CMR input vol has CMR data from three timesteps stacked as channel: t1,t1+t2/2,t2
        # transform only one timestep, mostly the first one
        transformed = TimeDistributed(st_lambda_layer, name='st_p2p')(keras.layers.Concatenate(axis=-1, name='cmr_flow_p2p')([input_tensor_raw, flows]))
        print('transformed_p2p:', transformed.shape)
        transformed_mask = TimeDistributed(st_mask_p2p_lambda_layer, name='st_p2p_msk')(
            keras.layers.Concatenate(axis=-1, name='msk_flow_p2p')([input_mask_tensor, flows]))

        if COMPOSE_CONSISTENCY:
            input_tensor_ed = stack_ed_lambda_layer(input_tensor_raw)
            # two options, either a 2nd unet for k2ed graph flow, or we re-use the existing one
            if dedicated_unet:
                config_temp['IMG_CHANNELS'] = input_tensor_ed.shape[-1]
                unet_ed = create_unet(config_temp, single_model=False)
            else:
                unet_ed = unet
                print('shared unet')
            pre_flows_p2ed = TimeDistributed(unet_ed, name='unet_ed')(input_tensor_ed)
            # composed flowfield should move each phase to ED
            flows_p2ed = TimeDistributed(conv_layer_p2ed, name='unet2flow_ed2p')(pre_flows_p2ed)
            flows_p2ed = add_zero_spatial_lambda_p2ed(flows_p2ed)
            transformed_p2ed = TimeDistributed(st_p2ed_lambda_layer, name='st_p2ed')(
                keras.layers.Concatenate(axis=-1, name='cmr_flow_p2ed')([input_tensor_raw, flows_p2ed]))
            transformed_mask_p2ed = TimeDistributed(st_mask_p2ed_lambda_layer, name='st_p2ed_mask')(
                keras.layers.Concatenate(axis=-1, name='mask_flow_p2ed')([input_mask_tensor, flows_p2ed]))
            transformed_p2ed = keras.layers.Lambda(lambda x: x, name='transformed_p2ed')(transformed_p2ed)
            transformed_masks = keras.layers.Concatenate(axis=-1, name='p2p_p2ed_mask')([transformed_mask_p2ed, transformed_mask])

            flows_p2ed = keras.layers.Lambda(lambda x: x, name='flowfield_p2ed')(flows_p2ed)
            print('comp transformed:', transformed_p2ed.shape)

        flow = keras.layers.Lambda(lambda x: x, name='flowfield_p2p')(flows)
        transformed_masks = keras.layers.Lambda(lambda x: x, name='transformed_mask')(transformed_masks)
        transformed = keras.layers.Lambda(lambda x: x, name='transformed_p2p')(transformed)

        outputs = [transformed, transformed_masks, flow]
        if COMPOSE_CONSISTENCY: outputs = [transformed_p2ed] + outputs + [flows_p2ed]

        model = Model(name='simpleregister', inputs=[input_tensor_raw, input_mask_tensor],
                      outputs=outputs)

        losses = {'transformed_p2p':image_loss_fn, 'transformed_mask':dice_coef_loss, 'flowfield_p2p':flow_reg_loss_fn}
        weights = {'transformed_p2p':image_loss_weight, 'transformed_mask':dice_loss_weight, 'flowfield_p2p':reg_loss_weight}

        if COMPOSE_CONSISTENCY:
            losses['transformed_p2ed'] = image_comp_loss_fn
            losses['flowfield_p2ed'] = flow_comp_reg_loss_fn
            weights['transformed_p2ed'] = image_loss_weight
            weights['flowfield_p2ed'] = reg_loss_weight
        print('losses: {}'.format(losses))
        print('weights: {}'.format(weights))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=losses,
                      loss_weights=weights)

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
