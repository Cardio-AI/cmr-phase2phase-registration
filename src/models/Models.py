import logging

from tensorflow.python.keras.layers import LSTM, Bidirectional

from src.models.KerasLayers import ConvEncoder, get_angle_tf, get_idxs_tf, get_centers_tf, ComposeTransform, \
    conv_layer_fn, ConvBlock

import sys
import numpy as np
import tensorflow
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import metrics as metr
from tensorflow.keras.layers import Dropout, BatchNormalization, TimeDistributed
import math

from src.models.Unets import create_unet
from src.utils import Metrics as own_metr

from src.models.ModelUtils import get_optimizer

sys.path.append('src/ext/neuron')
sys.path.append('src/ext/pynd-lib')
sys.path.append('src/ext/pytools-lib')
import src.ext.neuron.neuron.layers as nrn_layers

def create_PhaseRegressionModel(config, networkname='PhaseRegressionModel'):
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the "mirrored data"-paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
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
            gap_conv = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=(2, 1, 1), padding='valid',
                                              activation=activation, kernel_initializer=kernel_init)

            inputs_temporal = [tf.keras.layers.BatchNormalization()(elem) for elem in inputs_temporal]
            inputs_temporal = [tf.keras.layers.Dropout(rate=0.5)(elem) for elem in inputs_temporal]
            inputs_temporal = [gap_conv(vol) for vol in inputs_temporal]
        inputs_temporal = [gap(vol) for vol in inputs_temporal]
        inputs_temporal = tf.stack(inputs_temporal, axis=1)

        inputs = inputs_temporal
        print('Shape after GAP')
        print(inputs.shape)
        # 36, 256
        if add_bilstm:
            print('add a bilstm layer with: {} lstm units'.format(lstm_units))
            """inputs = tf.keras.layers.BatchNormalization()(inputs)
            inputs = tf.keras.layers.Dropout(rate=0.5)(inputs)
    
            onehot_pre = tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation=activation,
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
        onehot = tf.keras.layers.Conv1D(filters=PHASES, kernel_size=1, strides=1, padding='same', activation='linear',
                                        name='final_conv')(inputs)
        if final_activation == 'relu':
            onehot = tf.keras.activations.relu(onehot)
        elif final_activation == 'softmax':
            # axis -1 --> one class per timestep, as we repeat the phases its not possible to softmax the phase
            onehot = tf.keras.activations.softmax(onehot, axis=-1)
        elif final_activation == 'sigmoid':
            onehot = tf.keras.activations.sigmoid(onehot)
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


def create_PhaseRegressionModel_v2(config, networkname='PhaseRegressionModel'):
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the "mirrored data"-paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
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
        dim = config.get('DIM', [10, 224, 224])
        ndims = len(config.get('DIM', [10, 224, 224]))
        depth = config.get('DEPTH', 4)
        batchsize = config.get('BATCHSIZE', 8)
        add_bilstm = config.get('ADD_BILSTM', False)
        add_conv_bilstm = config.get('ADD_CONV_BILSTM', False)
        lstm_units = config.get('BILSTM_UNITS', 64)
        conv_lstm_units = config.get('CONV_BILSTM_UNITS', 64)
        add_vect_norm = config.get('ADD_VECTOR_NORM', False)
        add_vect_direction = config.get('ADD_VECTOR_DIRECTION', False)
        add_flows = config.get('ADD_FLOW', False)
        add_softmax = config.get('ADD_SOFTMAX', False)
        softmax_axis = config.get('SOFTMAX_AXIS', 1)
        image_loss_weight = config.get('IMAGE_LOSS_WEIGHT', 20)
        phase_loss_weight = config.get('PHASE_LOSS_WEIGHT', 1)
        flow_loss_weight = config.get('FLOW_LOSS_WEIGHT', 0.01)
        final_activation = config.get('FINAL_ACTIVATION', 'relu').lower()
        loss = config.get('LOSS', 'mse').lower()
        mask_loss = config.get('MASK_LOSS', False)
        downsample_flow_features = config.get('PRE_GAP_CONV', False)
        interp_method = 'linear'
        indexing = 'ij'
        # TODO: this parameter is also used by the generator to define the number of channels
        # here we stack the volume within the model manually, so keep that aligned!!!
        config['IMG_CHANNELS'] = 3  # we roll the temporal axis and stack t-1, t and t+1 along the last axis
        stack_axis = 1

        ############################# definition of the layers ######################################
        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]
        # start with very small deformation
        Conv = getattr(KL, 'Conv{}D'.format(ndims))
        Conv_layer = Conv(ndims, kernel_size=3, padding='same',
                          kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5),
                          name='unet2flow')

        unet = create_unet(config, single_model=False, networkname='3D-Unet')
        st_layer = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True,
                                                 name='deformable_layer')
        st_lambda_layer = tf.keras.layers.Lambda(
            lambda x: st_layer([x[..., 0:1], x[..., 1:]]), name='deformable_lambda_layer')

        gap = tensorflow.keras.layers.GlobalAveragePooling3D(name='GAP_3D_Layer')
        roll_concat_lambda_layer = tf.keras.layers.Lambda(lambda x:
                                                          tf.keras.layers.Concatenate(axis=-1, name='stack_with_moved')(
                                                              [tf.roll(x, shift=1, axis=stack_axis), x,
                                                               tf.roll(x, shift=-1, axis=stack_axis)]))

        norm_lambda = tf.keras.layers.Lambda(
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
        centers = c - idx
        centers_tensor = centers[tf.newaxis, ...]
        flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')

        forward_conv_lstm_layer = tf.keras.layers.ConvLSTM2D(filters=conv_lstm_units,
                                                             kernel_size=3,
                                                             strides=1,
                                                             padding='valid',
                                                             return_sequences=True,
                                                             dropout=0.5,
                                                             name='forward_conv_LSTM')
        backward_conv_lstm_layer = tf.keras.layers.ConvLSTM2D(filters=conv_lstm_units,
                                                              kernel_size=3,
                                                              strides=1,
                                                              padding='valid',
                                                              return_sequences=True,
                                                              go_backwards=True,
                                                              dropout=0.5,
                                                              name='backward_conv_LSTM')
        bi_conv_lstm_layer = Bidirectional(forward_conv_lstm_layer, backward_layer=backward_conv_lstm_layer)

        forward_layer = LSTM(lstm_units, return_sequences=True, dropout=0.3, name='forward_LSTM')
        backward_layer = LSTM(lstm_units, return_sequences=True, dropout=0.3, go_backwards=True,
                              name='backward_LSTM')  # maybe change to tanh
        bi_lstm_layer = Bidirectional(forward_layer, backward_layer=backward_layer)

        # How to downscale the in-plane/spatial resolution?
        # 1st idea: apply conv layers with a stride
        # b, t, 16, 64, 64, 3/4
        # conv with: n times 4,4,4 filters, valid/no border padding and a stride of 4
        # b, t, 1, 1, 1, n
        # 2nd idea: GAP with/without pre-conv layer which extracts motion features into the channels
        # 3rd idea use the tft.pca module to transform the downstream.
        # This would reduce the dimension of input vectors to output_dim in a way that retains the maximal variance

        downsamples = []
        d_rate = 0.2
        filters_ = 16
        #  b, t, 4, 16, 16, n
        # two times conv with: n times 4,4,4 filters, valid/no border padding and a stride of 4
        # b, t, 1, 4, 4, n
        # conv with: n times 4,4,4 filters, valid/no border padding and a stride of 4
        # how often can we downsample the inplane/spatial resolution until we reach 1
        # n = ln(1/x)/ln0,5

        n = int(math.log(1 / dim[-1]) / math.log(0.5))
        z = int(math.log(1 / dim[0]) / math.log(0.5))
        for i in range(n):
            if i < z:
                #downsamples.append(Dropout(d_rate))
                downsamples.append(
                    # Deformable conv
                    Conv(filters=filters_, kernel_size=3, padding='same', strides=1,
                         kernel_initializer=kernel_init,
                         activation=activation,
                         name='downsample_{}'.format(i)))
                downsamples.append(tf.keras.layers.MaxPool3D(pool_size=2,padding='same'))
                filters_ = filters_ * 2
            else:  # stop to down-sample the spatial resolution, continue with 2D conv
                downsamples.append(
                    tf.keras.layers.Conv2D(filters=filters_, kernel_size=(3, 3), padding='same', strides=1,
                                           kernel_initializer=kernel_init,
                                           activation=activation,
                                           name='downsample_{}'.format(i))
                )
                downsamples.append(tf.keras.layers.MaxPool3D(pool_size=(1,2,2), padding='same'))
            downsamples.append(BatchNormalization(axis=-1))

        downsamples = downsamples[:-1]  # remove last BN layer

        downsample = tf.keras.Sequential(layers=downsamples, name='downsample_inplane_and_spatial')
        final_onehot_conv = tf.keras.layers.Conv1D(filters=PHASES, kernel_size=1, strides=1, padding='same',
                                                   kernel_initializer=kernel_init, activation=final_activation,
                                                   name='pre_onehot')

        ##################################### Layer definition end ##############################################

        # unstack along t yielding a list of 3D volumes
        # roll t-1, t and t+1 (axis 1) and stack on axis -1, yielding a temporal window for the 3D conv net

        print('Shape Input Tensor: {}'.format(input_tensor.shape))
        inputs_spatial_stacked = roll_concat_lambda_layer(input_tensor)
        print('Shape rolled and stacked: {}'.format(inputs_spatial_stacked.shape))
        pre_flows = TimeDistributed(unet)(inputs_spatial_stacked)
        print('Unet output shape: {}'.format(pre_flows.shape))
        flows = TimeDistributed(Conv_layer)(pre_flows)
        print('Flowfield shape: {}'.format(flows.shape))
        transformed = TimeDistributed(st_lambda_layer)(tf.keras.layers.Concatenate(axis=-1)([input_tensor, flows]))
        print('Transformed shape : {}'.format(transformed.shape))

        if add_vect_norm and add_flows:
            # add the magnitude as fourth channel
            tensor_magnitude = TimeDistributed(norm_lambda)(flows)
            flow_features = tf.keras.layers.Concatenate(axis=-1)([flows, tensor_magnitude])
        elif add_vect_norm:
            # add the magnitude as fourth channel
            tensor_magnitude = TimeDistributed(norm_lambda)(flows)
            flow_features = tensor_magnitude
        elif add_flows:
            flow_features = flows
        print('Inkl norm shape: {}'.format(flow_features.shape))

        if add_vect_direction:
            directions = TimeDistributed(flow2direction_lambda)(flows)
            print('directions shape: {}'.format(directions.shape))
            flow_features = tf.keras.layers.Concatenate(axis=-1)([flow_features, directions]) # encode the spatial location of each vector
            # add the location tensor as further channel
            #flow_features = tf.keras.layers.Concatenate(axis=-1)([flow_features, tf.tile(centers_tensor[tf.newaxis,...],multiples=[1,flow_features.shape[1],1,1,1,1])])
        print('flow features inkl directions shape: {}'.format(flow_features.shape))

        # Apply an Bidirectional convLstm layer before downsampling
        if add_conv_bilstm:
            flow_features = tf.transpose(flow_features, perm=[0, 2, 1, 3, 4, 5])
            print('transposed: {}'.format(flow_features.shape))
            flow_features = TimeDistributed(bi_conv_lstm_layer)(flow_features)
            flow_features = tf.transpose(flow_features, perm=[0, 2, 1, 3, 4, 5])
        print('flow features after conv lstm: {}'.format(flow_features.shape))

        # down-sample the flow in-plane
        # Build an encoder with n times conv+relu+maxpool+bn-blocks
        if downsample_flow_features:
            flow_features = TimeDistributed(downsample)(flow_features)
        else: # use a gap3D layer
            flow_features = TimeDistributed(gap)(flow_features)
        flow_features = tf.keras.layers.Reshape(target_shape=(flow_features.shape[1], flow_features.shape[-1]))(
            flow_features)

        if add_bilstm:
            print('Shape before LSTM layers: {}'.format(flow_features.shape))
            flow_features = bi_lstm_layer(flow_features)
            print('Shape after LSTM layers: {}'.format(flow_features.shape))

        # input (t,encoding) output (t,5)
        # 36, 5
        onehot = final_onehot_conv(flow_features)
        print('Shape after final conv layer: {}'.format(onehot.shape))
        # add empty tensor with one-hot shape to align with gt
        if add_softmax: onehot = tf.keras.activations.softmax(onehot, axis=softmax_axis)

        zeros = tf.zeros_like(onehot, name='zero_padding')
        onehot = tf.stack([onehot, zeros], axis=1, name='extend_onehot_by_zeros')

        # define the model output names
        onehot = tf.keras.layers.Activation('linear', name='onehot')(onehot)
        transformed = tf.keras.layers.Activation('linear', name='transformed')(transformed)
        flows = tf.keras.layers.Activation('linear', name='flows')(flows)

        outputs = [onehot, transformed, flows]

        if loss == 'cce':
            losses = [own_metr.CCE(masked=mask_loss, smooth=0.8, transposed=False)]
        elif loss == 'meandiff':
            losses = [own_metr.Meandiff_loss()]
        elif loss == 'msecce':
            print(loss)
            losses = [own_metr.MSE(), own_metr.CCE(masked=mask_loss, smooth=0.8, transposed=False)]
            print(losses)
        else:  # default fallback --> MSE - works the best
            from tensorflow.keras.losses import mse
            from src.utils.Metrics import Grad, MSE_
            losses = {
                # changed!!!!
                # 'onehot': own_metr.Meandiff_loss(),
                # 'onehot': own_metr.CCE(masked=mask_loss, smooth=0.8, transposed=False),
                'onehot': own_metr.MSE(masked=mask_loss),
                'transformed': MSE_().loss,
                'flows': Grad('l2').loss}
            weights = {
                'onehot': phase_loss_weight,
                'transformed': image_loss_weight,
                'flows': flow_loss_weight}

        print('added loss: {}'.format(loss))
        model = Model(inputs=[input_tensor], outputs=outputs, name=networkname)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('LEARNING_RATE', 0.001)),
            loss=losses,
            loss_weights=weights,
            metrics={
                'onehot': own_metr.meandiff,
                # 'transformed': MSE_().loss,
                # 'flows': Grad('l2').loss
            }
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

        from tensorflow.keras.losses import mse
        from src.utils.Metrics import Grad, MSE_

        losses = [MSE_().loss, Grad('l2').loss]
        weights = [image_loss_weight, reg_loss_weight]
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=losses,
                      loss_weights=weights)

    return model


def create_RegistrationModel_inkl_mask(config):
    """
    A registration wrapper for 3D image2image registration
    """
    import random
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

        # input vol with timesteps, z, x, y, c -> =number of input timesteps
        input_tensor = Input(shape=(T_SHAPE, *input_shape, config.get('IMG_CHANNELS',1)))
        input_mask_tensor = Input(shape=(T_SHAPE, *input_shape, config.get('IMG_CHANNELS', 1)))
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
        st_mask_layer = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True,
                                                      name='deformable_mask_layer')

        unet = create_unet(config, single_model=False)

        input_vols = tf.unstack(input_tensor, axis=1)
        input_mask_vols = tf.unstack(input_mask_tensor, axis=1)
        print(input_vols[0].shape)

        indicies = list(tf.range(len(input_vols)))
        zipped = list(zip(input_vols, indicies))
        # random.shuffle(zipped)
        input_vols_shuffled, indicies = zip(*zipped)
        pre_flows = [unet(vol) for vol in input_vols_shuffled]
        flows = [Conv_layer(vol) for vol in pre_flows]
        flows, _ = zip(*sorted(zip(flows, indicies), key=lambda tup: tup[1]))

        # Each CMR input vol has CMR data from three timesteps stacked as channel: t1,t1+t2/2,t2
        # transform only one timestep, mostly the first one
        transformed = [st_layer([input_vol[..., take_t_elem][..., tf.newaxis], flow]) for input_vol, flow in
                       zip(input_vols, flows)]

        # Remove the Z flow when transforming the masks, as we provide only spare masks,
        # moving a mask in z results in a transformed mask at Z where we dont have a mask in the GT
        # The dice loss, will than backpropagate not to move in Z at all, as this will reduce the dice loss
        #empty_z_flow = tf.zeros_like(flows[0])
        #mod_mask_flows = [tf.concat([empty_z_flow[...,0:1], flow[...,1:]], axis=-1) for flow in flows]
        transformed_mask = [st_mask_layer([input_vol[..., take_t_elem][..., tf.newaxis], flow]) for input_vol, flow in
                            zip(input_mask_vols, flows)]

        transformed = tf.stack(transformed, axis=1)
        transformed_mask = tf.stack(transformed_mask, axis=1)
        flow = tf.stack(flows, axis=1)

        if COMPOSE_CONSISTENCY:
            # composed flowfield should move each phase to ED
            comp = create_dense_compose(config)
            flows_composed = comp(flow)
            flows_composed = tf.unstack(flows_composed, axis=1)
            # they should transform each timestep to ED
            comp_transformed = [st_layer([input_vol[..., take_t_elem][..., tf.newaxis], flow]) for input_vol, flow in
                           zip(input_vols, flows_composed)]
            comp_transformed = tf.stack(comp_transformed, axis=1)
            comp_transformed = tf.keras.layers.Lambda(lambda x: x, name='comp_transformed')(comp_transformed)

        flow = tf.keras.layers.Lambda(lambda x : x, name='flowfield')(flow)
        transformed_mask = tf.keras.layers.Lambda(lambda x: x, name='transformed_mask')(transformed_mask)
        transformed = tf.keras.layers.Lambda(lambda x: x, name='transformed')(transformed)

        outputs = [transformed, transformed_mask, flow]
        if COMPOSE_CONSISTENCY: outputs = [comp_transformed] + outputs

        model = Model(name='simpleregister', inputs=[input_tensor, input_mask_tensor, input_tensor_empty],
                      outputs=outputs)

        from tensorflow.keras.losses import mse
        from src.utils.Metrics import Grad, MSE_
        from src.utils.Metrics import dice_coef_loss

        losses = [MSE_().loss, dice_coef_loss, Grad('l2').loss]
        if COMPOSE_CONSISTENCY: losses = [MSE_().loss] + losses
        weights = [image_loss_weight, dice_loss_weight, reg_loss_weight]
        if COMPOSE_CONSISTENCY: weights = [image_loss_weight] + weights
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=losses,
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
    :return: compiled tf.keras model
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
    :return: compiled tf.keras model
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

        # we need to reverse the transforms as we register from t+1 to t.
        # we need to provide the order in which we would apply the compose transforms
        # ED, MS, ES, PF, MD
        # flows:
        # 0= MS->ED, 1=ES->MS, 2=PF->ES, 3=MD->PF, 4=ED->MD
        #e.g. for compose:
        # MS->ED = [0]
        # ES->ED = [1,0]
        # PF->ED = [2,1,0]
        # list(reversed())
        if reverse:
            y = [ComposeTransform(interp_method='linear', shift_center=True, indexing=indexing,
                                  name='Compose_transform{}'.format(i))(list(reversed(flows[:i]))) for i in range(2, len(flows) + 1)]
        else:
            y = [ComposeTransform(interp_method='linear', shift_center=True, indexing=indexing,
                              name='Compose_transform{}'.format(i))(flows[:i]) for i in range(2, len(flows) + 1)]
        y = tf.stack([flows[0], *y], axis=1)

        model = Model(inputs=[inputs], outputs=[y], name=networkname)

        return model
