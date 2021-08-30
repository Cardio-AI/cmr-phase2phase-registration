import logging

from tensorflow.python.keras.layers import LSTM, Bidirectional

from src.models.KerasLayers import ConvEncoder
import sys
import numpy as np
import tensorflow
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import metrics as metr

from src.models.Unets import create_unet
from src.utils import Metrics as own_metr

from src.models.ModelUtils import get_optimizer

sys.path.append('src/ext/neuron')
sys.path.append('src/ext/pynd-lib')
sys.path.append('src/ext/pytools-lib')
import src.ext.neuron.neuron.layers as nrn_layers
#import neurite as ne


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
            forward_layer = LSTM(lstm_units,return_sequences=True)
            backward_layer = LSTM(lstm_units, activation='tanh', return_sequences=True,go_backwards=True) # maybe change to tanh
            inputs = Bidirectional(forward_layer, backward_layer=backward_layer,input_shape=inputs.shape)(inputs)

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
            losses = [own_metr.CCE(masked=mask_loss, smooth=0.8,transposed=False)]
        elif loss == 'meandiff':
            losses = [own_metr.Meandiff_loss()]
        elif loss == 'msecce':
            print(loss)
            #own_metr.CCE(masked=False,smooth=0.8,transposed=False)
            losses = [own_metr.MSE(), own_metr.CCE(masked=mask_loss, smooth=0.8,transposed=False)]
            print(losses)
        else: # default fallback --> MSE - works the best
            losses = [own_metr.MSE(masked=mask_loss)]

        print('added loss: {}'.format(loss))
        model = Model(inputs=[input_tensor], outputs=outputs, name=networkname)
        model.compile(
            optimizer=get_optimizer(config, networkname),
            loss=losses,
            #metrics=[own_metr.mse_wrapper, own_metr.ca_wrapper, own_metr.meandiff] #
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
        lstm_units = config.get('BILSTM_UNITS', 64)
        final_activation = config.get('FINAL_ACTIVATION', 'relu').lower()
        loss = config.get('LOSS', 'mse').lower()
        mask_loss = config.get('MASK_LOSS', False)
        pre_gap_conv = config.get('PRE_GAP_CONV', False)
        interp_method = 'linear'
        indexing = 'ij'
        # TODO: this parameter is also used by the generator to define the number of channels
        # here we stack the volume within the model manually, so keep that aligned!!!
        #config['IMG_CHANNELS'] = 3
        ############################# definition of the layers ######################################
        # increase the dropout through the layer depth
        dropouts = list(np.linspace(drop_1, drop_3, depth))
        dropouts = [round(i, 1) for i in dropouts]
        # start with very small deformation
        Conv = getattr(KL, 'Conv{}D'.format(ndims))
        Conv_layer = Conv(ndims, kernel_size=3, padding='same',
                          kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5),
                          name='unet2flow')

        spatial_encoder = ConvEncoder(activation=activation,
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
        config['IMG_CHANNELS'] = 3 # we roll the temporal axis and stack t-1, t and t+1 along the last axis
        unet = create_unet(config, single_model=False, networkname='3D-Unet')
        st_layer = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True,
                                                 name='deformable_layer')

        gap = tensorflow.keras.layers.GlobalAveragePooling3D(name='GAP_3D_Layer')
        concat_layer = tf.keras.layers.Concatenate(axis=-1, name='stack_with_moved')

        norm_lambda = tf.keras.layers.Lambda(
            lambda x: tf.norm(x, ord='euclidean', axis=-1, keepdims=True, name='flow2norm'), name='flow2norm')
        concat_lambda = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1, name='extend_flow_with_norm'),
                                               name='stack_flow_with_norm')
        concat_lambda2 = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1, name='extend_flow_with_direction'),
                                               name='stack_flow_with_direction')
        flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[0],x[1]), name='flow2direction')

        # How to downscale the in-plane and spatial resolution?
        # 1st idea: apply conv layers with a stride
        # b, t, 16, 64, 64, 3/4
        # conv with: n times 4,4,4 filters, valid/no border padding and a stride of 4
        # b, t, 1, 1, 1, n

        # global average pooling for T x 3D vols
        #inputs_spatial = [gap(vol) for vol in inputs_spatial]
        # 2nd idea: GAP with/without pre-conv layer which extracts motion features into the channels
        # 3rd idea use the tft.pca module to transform the downstream.
        # This would reduce the dimension of input vectors to output_dim in a way that retains the maximal variance
        from src.models.KerasLayers import conv_layer_fn, ConvBlock
        from tensorflow.keras.layers import Dropout, BatchNormalization
        downsamples = []
        d_rate = 0.2
        filters_ = 16
        #  b, t, 4, 16, 16, n
        # two times conv with: n times 4,4,4 filters, valid/no border padding and a stride of 4
        # b, t, 1, 4, 4, n
        # conv with: n times 4,4,4 filters, valid/no border padding and a stride of 4
        # how often can we downsample the inplane and spatial resolution until we reach 1
        # n = ln(1/x)/ln0,5
        import math
        n = int(math.log(1/dim[-1])/math.log(0.5))
        z = int(math.log(1/dim[0])/math.log(0.5))
        for i in range(n):
            #downsamples.append(Dropout(d_rate))
            if i < z:
                downsamples.append(
                    Conv(filters=filters_, kernel_size=3, padding='same', strides=2,
                         kernel_initializer=kernel_init,
                         activation=activation,
                         name='downsample_{}'.format(i)))
                filters_ = filters_ * 2
            else: # stop to downsample the spatial resolution
                downsamples.append(
                    Conv(filters=filters_, kernel_size=(1, 3, 3), padding='same', strides=(1, 2, 2),
                         kernel_initializer=kernel_init,
                         activation=activation,
                         name='downsample_{}'.format(i)))
            downsamples.append(BatchNormalization(axis=-1))
        downsamples = downsamples[:-1] # remove last BN layer

        downsample = tf.keras.Sequential(layers=downsamples, name='downsample_inplane_and_spatial')
        final_onehot_conv = tf.keras.layers.Conv1D(filters=PHASES, kernel_size=1, strides=1, padding='same', kernel_initializer=kernel_init, activation=final_activation,
                                        name='pre_onehot')
        ##################################### Layer definition ##############################################

        import random
        # unstack along t yielding a list of 3D volumes
        # stack t-1, t and t+1 along the channels
        unet_axis = 1
        stack_axis = 1
        #inputs_spatial_stacked = input_tensor
        print('Shape Input Tensor: {}'.format(input_tensor.shape))
        inputs_spatial_stacked = concat_layer([tf.roll(input_tensor, shift=1, axis=stack_axis), input_tensor, tf.roll(input_tensor, shift=-1, axis=stack_axis)])
        print('Shape rolled and stacked: {}'.format(inputs_spatial_stacked.shape))
        inputs_spatial_unstacked = tf.unstack(inputs_spatial_stacked, axis=unet_axis, name='split_into_t_times_3D_vols')
        print('Shape after unstacking: {} x {}'.format(len(inputs_spatial_unstacked), inputs_spatial_unstacked[0].shape))
        # first tests without shuffle, later we can add it, it seems to drop the train/val gap
        '''indicies = list(tf.range(len(inputs_spatial_unstacked)))
        zipped = list(zip(inputs_spatial_unstacked, indicies))
        random.shuffle(zipped)
        inputs_spatial_unstacked, indicies = zip(*zipped)'''
        # feed the T x 3D volumes into the spatial encoder (3D conv)
        pre_flows = [unet(vol) for vol in inputs_spatial_unstacked]
        #pre_flows, _ = zip(*sorted(zip(pre_flows, indicies), key=lambda tup: tup[1]))
        flows= [Conv_layer(vol) for vol in pre_flows]

        flows_stacked = tf.stack(flows, axis=unet_axis, name='stack_flows')
        pre_flows_stacked = tf.stack(pre_flows, axis=unet_axis, name='stack_preflows')

        transformed = [st_layer([input_vol, flow]) for input_vol, flow in
                       zip(tf.unstack(input_tensor, axis=1), flows)]
        print('Flowfield shape: {}'.format(flows[0].shape))

        # add the magnitude as fourth channel
        tensor_magnitude = [norm_lambda(vol) for vol in flows]
        flow_features = flows
        #flow_features = [concat_lambda([flow,norm]) for flow,  norm in zip(flows, tensor_magnitude)]
        print('inkl norm shape: {}'.format(flow_features[0].shape))
        #calculate the flowfield direction compared to a displacment field which always points to the center
        def get_angle_tf(a, b):
            # this should work for batches of n-dimensional vectors
            # α = arccos[(a · b) / (|a| * |b|)]
            # |v| = √(x² + y² + z²)
            """
            in 3D space
            If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
            α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
            """
            import tensorflow as tf
            import math as m
            pi = tf.constant(m.pi)
            #a, b = tf.convert_to_tensor(a, dtype=tf.float32), tf.convert_to_tensor(b, dtype=tf.float32)
            inner = tf.einsum('...i,...i->...', a, b)
            norms = (tf.norm(a, axis=-1) * tf.norm(b, axis=-1))  # [...,None]
            cos = inner / (norms + sys.float_info.epsilon)
            rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0))
            # rad2deg conversion
            deg = rad * (180.0/pi)
            return deg[...,tf.newaxis]

        # returns a matrix with the indicies as values, similar to np.indicies
        get_idxs_tf = lambda x: tf.cast(
            tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
            tf.float32)
        # returns a matrix with vectors pointing to the center
        get_centers_tf = lambda x: tf.cast(
            tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...], (x[0], x[1], x[2], 1)), tf.float32)

        # get a tensor with vectors pointing to the center
        # get idxs of one 3D
        # get a tensor with the same shape as the flowfield with the repeated vector toward the center
        # calculate the difference, which should yield a 3D tensor with vectors pointing to the center
        # tile/repeat this v_center vol along the temporal and batch axis
        # calculate the angle of each voxel between the tiled v_center tensor and the displacement tensor
        # concat this tensor as additional feature to the last axis of flow_features
        flow_shape = tf.shape(flows[0])
        idx = get_idxs_tf(dim)
        c = get_centers_tf(dim)
        centers = c - idx

        centers_tensor = tf.tile(centers[tf.newaxis,...], (flow_shape[0], 1,1,1,1))

        directions = [flow2direction_lambda([flow, centers_tensor]) for flow in flows]
        print('directions shape: {}'.format(directions[0].shape))

        # make unit vectors

        #flow_features = [concat_lambda2([flow_f, angles]) for flow_f, angles in zip(flow_features, directions)]
        # calculate the direction between the flowfield and the centerflow
        print('flow features inkl directions shape: {}'.format(flow_features[0].shape))
        flow_features = [downsample(vol) for vol in flow_features]

        flow_features = tf.stack(flow_features, axis=1, name='Stack_flow_features')
        flow_features = tf.keras.layers.Reshape(target_shape=(flow_features.shape[1],flow_features.shape[-1]))(flow_features)


        if add_bilstm:
            print('add a bilstm layer with: {} lstm units'.format(lstm_units))
            print('Shape before LSTM layers: {}'.format(flow_features.shape))
            forward_layer = LSTM(lstm_units, return_sequences=True, name='forward_LSTM')
            backward_layer = LSTM(lstm_units, activation='tanh', return_sequences=True,
                                  go_backwards=True, name='backward_LSTM')  # maybe change to tanh
            flow_features = Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=flow_features.shape)(flow_features)
            print('Shape after bilstm layer: {}'.format(flow_features.shape))

        # input (36,encoding) output (36,5)
        onehot = final_onehot_conv(flow_features)

        # 36, 5
        print('Shape after final conv layer: {}'.format(onehot.shape))
        transformed = tf.stack(transformed, axis=1, name='stacked_transformed')
        # add empty tensor with one-hot shape to align with gt
        zeros = tf.zeros_like(onehot, name='zero_padding')
        onehot = tf.stack([onehot, zeros], axis=1, name='extend_onehot_by_zeros')

        onehot = tf.keras.layers.Activation('linear', name='onehot')(onehot)
        transformed = tf.keras.layers.Activation('linear', name='transformed')(transformed)
        flows = tf.keras.layers.Activation('linear', name='flows')(flows_stacked)

        outputs = [onehot, transformed, flows]

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
            from tensorflow.keras.losses import mse
            from src.utils.Metrics import Grad, MSE_
            losses = {
                'onehot': own_metr.MSE(masked=mask_loss),
                'transformed': MSE_().loss,
                'flows': Grad('l2').loss}

            weights = {
                'onehot': 1,
                'transformed': 20,
                'flows': 0.01}

        print('added loss: {}'.format(loss))
        model = Model(inputs=[input_tensor], outputs=outputs, name=networkname)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('LEARNING_RATE', 0.001)),
            loss=losses,
            loss_weights=weights,
            # metrics=[own_metr.mse_wrapper, own_metr.ca_wrapper, own_metr.meandiff] #
            metrics={
                'onehot': own_metr.meandiff,
                'transformed': MSE_().loss,
                'flows': Grad('l2').loss
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


        input_tensor = Input(shape=(T_SHAPE, *input_shape, config.get('IMG_CHANNELS', 1))) # input vol with timesteps, z, x, y, c -> =number of input timesteps
        input_tensor_empty = Input(shape=(T_SHAPE, *input_shape, 3)) # empty vector field
        # define standard values according to the convention over configuration paradigm

        ndims = len(input_shape)
        indexing = 'ij'
        interp_method = 'linear'
        Conv = getattr(KL, 'Conv{}D'.format(ndims))
        take_t_elem = config.get('INPUT_T_ELEM', 0)

        # start with very small deformation
        Conv_layer = Conv(ndims, kernel_size=3, padding='same',
                    kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5), name='unet2flow')
        st_layer = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True, name='deformable_layer')

        unet = create_unet(config, single_model=False)

        input_vols = tf.unstack(input_tensor, axis=1)
        print(input_vols[0].shape)
        import random
        indicies = list(tf.range(len(input_vols)))
        zipped = list(zip(input_vols, indicies))
        random.shuffle(zipped)
        input_vols_shuffled, indicies = zip(*zipped)
        #input_vols_shuffled = input_vols
        pre_flows = [unet(vol) for vol in input_vols_shuffled]
        flows= [Conv_layer(vol) for vol in pre_flows]
        flows, _ = zip(*sorted(zip(flows, indicies), key=lambda tup: tup[1]))

        transformed = [st_layer([input_vol[...,take_t_elem][...,tf.newaxis], flow]) for input_vol, flow in zip(input_vols, flows)]
        transformed = tf.stack(transformed, axis=1)
        flow = tf.stack(flows, axis=1)

        outputs = [transformed, flow]

        model = Model(name='simpleregister', inputs=[input_tensor, input_tensor_empty], outputs=outputs)

        from tensorflow.keras.losses import mse
        from src.utils.Metrics import Grad, MSE_

        losses = [MSE_().loss, Grad('l2').loss]
        weights = [image_loss_weight, reg_loss_weight]
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=losses, loss_weights=weights)

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


        input_tensor = Input(shape=(T_SHAPE, *input_shape, config.get('IMG_CHANNELS', 1))) # input vol with timesteps, z, x, y, c -> =number of input timesteps
        input_mask_tensor = Input(shape=(T_SHAPE, *input_shape, config.get('IMG_CHANNELS', 1)))
        input_tensor_empty = Input(shape=(T_SHAPE, *input_shape, 3)) # empty vector field
        # define standard values according to the convention over configuration paradigm

        ndims = len(input_shape)
        indexing = 'ij'
        interp_method = 'linear'
        Conv = getattr(KL, 'Conv{}D'.format(ndims))
        take_t_elem = config.get('INPUT_T_ELEM', 0)

        # start with very small deformation
        Conv_layer = Conv(ndims, kernel_size=3, padding='same',
                    kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5), name='unet2flow')
        st_layer = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True, name='deformable_layer')
        st_mask_layer = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True,
                                                 name='deformable_mask_layer')

        unet = create_unet(config, single_model=False)

        input_vols = tf.unstack(input_tensor, axis=1)
        input_mask_vols = tf.unstack(input_mask_tensor, axis=1)
        print(input_vols[0].shape)

        indicies = list(tf.range(len(input_vols)))
        zipped = list(zip(input_vols, indicies))
        #random.shuffle(zipped)
        input_vols_shuffled, indicies = zip(*zipped)
        pre_flows = [unet(vol) for vol in input_vols_shuffled]
        flows= [Conv_layer(vol) for vol in pre_flows]
        flows, _ = zip(*sorted(zip(flows, indicies), key=lambda tup: tup[1]))

        # transform only one timestep, mostly the first one
        transformed = [st_layer([input_vol[...,take_t_elem][...,tf.newaxis], flow]) for input_vol, flow in zip(input_vols, flows)]
        transformed = tf.stack(transformed, axis=1)
        transformed_mask = [st_mask_layer([input_vol[..., take_t_elem][..., tf.newaxis], flow]) for input_vol, flow in
                       zip(input_mask_vols, flows)]
        transformed_mask = tf.stack(transformed_mask, axis=1)

        flow = tf.stack(flows, axis=1)

        outputs = [transformed, transformed_mask, flow]

        model = Model(name='simpleregister', inputs=[input_tensor, input_mask_tensor, input_tensor_empty], outputs=outputs)

        from tensorflow.keras.losses import mse
        from src.utils.Metrics import Grad, MSE_
        from src.utils.Metrics import dice_coef_loss

        losses = [MSE_().loss, dice_coef_loss, Grad('l2').loss]
        weights = [image_loss_weight, dice_loss_weight, reg_loss_weight]
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=losses, loss_weights=weights)

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

        inputs = Input((5,*config.get('DIM', [10, 224, 224]), 3))
        input_displacement = Input((*config.get('DIM', [10, 224, 224]), 3))
        indexing = config.get('INDEXING', 'ij')

        # warp the source with the flow
        flows = tf.unstack(inputs, axis=1)

        y = [ComposeTransform(interp_method='linear', shift_center=True, indexing=indexing, name='Compose_transform{}'.format(i))(flows[:i]) for i in range(2,len(flows)+1)]
        y = tf.stack([flows[0],*y], axis=1)

        model = Model(inputs=[inputs], outputs=[y], name=networkname)

        return model


class ComposeTransform(tf.keras.layers.Layer):
    """
    Composes a single transform from a series of transforms.
    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:
    T = ComposeTransform()([A, B, C])
    """

    def __init__(self, interp_method='linear', shift_center=True, indexing='ij', **kwargs):
        """
        Parameters:
            shape: Target shape of dense shift.
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            shift_center: Shift grid to image center.
            indexing: Must be 'xy' or 'ij'.
        """
        self.interp_method = interp_method
        self.shift_center = shift_center
        self.indexing = indexing
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'shift_center': self.shift_center,
            'indexing': self.indexing,
        })
        return config

    def build(self, input_shape, **kwargs):

        # sanity check on the inputs
        if not isinstance(input_shape, (list, tuple)):
            raise Exception('ComposeTransform must be called for a list of transforms.')
        if len(input_shape) < 2:
            raise ValueError('ComposeTransform input list size must be greater than 1.')

        # determine output transform type
        dense_shape = next((t for t in input_shape if not is_affine_shape(t[1:])), None)
        if dense_shape is not None:
            # extract shape information from the dense transform
            self.outshape = (input_shape[0], *dense_shape)
        else:
            # extract dimension information from affine
            ndims = input_shape[0][-1] - 1
            self.outshape = (input_shape[0], ndims, ndims + 1)

    def call(self, transforms):
        """
        Parameters:
            transforms: List of affine or dense transforms to compose.
        """
        compose_ = lambda trf: compose(trf, interp_method=self.interp_method,shift_center=self.shift_center, indexing=self.indexing)
        return tf.map_fn(compose_, transforms, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return self.outshape

def compose(transforms, interp_method='linear', shift_center=True, indexing='ij'):
    """
    Compose a single transform from a series of transforms.
    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:
    T = compose([A, B, C])
    Parameters:
        transforms: List of affine and/or dense transforms to compose.
        interp_method: Interpolation method. Must be 'linear' or 'nearest'.
        shift_center: Shift grid to image center.
        indexing: Must be 'xy' or 'ij'.
    Returns:
        Composed affine or dense transform.
    """
    if indexing != 'ij':
        raise ValueError('Compose transform only supports ij indexing')

    if len(transforms) < 2:
        raise ValueError('Compose transform list size must be greater than 1')

    def ensure_dense(trf, shape):
        if is_affine_shape(trf.shape):
            return affine_to_dense_shift(trf, shape, shift_center=shift_center, indexing=indexing)
        return trf

    def ensure_square_affine(matrix):
        if matrix.shape[-1] != matrix.shape[-2]:
            return make_square_affine(matrix)
        return matrix

    curr = transforms[-1]
    for nxt in reversed(transforms[:-1]):
        # check if either transform is dense
        found_dense = next((t for t in (nxt, curr) if not is_affine_shape(t.shape)), None)
        if found_dense is not None:
            # compose dense warps
            shape = found_dense.shape[:-1]
            nxt = ensure_dense(nxt, shape)
            curr = ensure_dense(curr, shape)
            curr = curr + transform(nxt, curr, interp_method=interp_method, indexing=indexing)
        else:
            # compose affines
            nxt = ensure_square_affine(nxt)
            curr = ensure_square_affine(curr)
            curr = tf.linalg.matmul(nxt, curr)[:-1]

    return curr

def make_square_affine(mat):
    """
    Converts a [N, N+1] affine matrix to square shape [N+1, N+1].
    Parameters:
        mat: Affine matrix of shape [..., N, N+1].
    """
    validate_affine_shape(mat.shape)
    bs = mat.shape[:-2]
    zeros = tf.zeros((*bs, 1, mat.shape[-2]), dtype=mat.dtype)
    one = tf.ones((*bs, 1, 1), dtype=mat.dtype)
    row = tf.concat((zeros, one), axis=-1)
    mat = tf.concat([mat, row], axis=-2)
    return mat

def affine_to_dense_shift(matrix, shape, shift_center=True, indexing='ij'):
    """
    Transforms an affine matrix to a dense location shift.
    Algorithm:
        1. Build and (optionally) shift grid to center of image.
        2. Apply affine matrix to each index.
        3. Subtract grid.
    Parameters:
        matrix: affine matrix of shape (N, N+1).
        shape: ND shape of the target warp.
        shift_center: Shift grid to image center.
        indexing: Must be 'xy' or 'ij'.
    Returns:
        Dense shift (warp) of shape (*shape, N).
    """
    import neurite as ne

    if isinstance(shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        shape = shape.as_list()

    if matrix.dtype != 'float32':
        matrix = tf.cast(matrix, 'float32')

    # check input shapes
    ndims = len(shape)
    if matrix.shape[-1] != (ndims + 1):
        matdim = matrix.shape[-1] - 1
        raise ValueError(f'Affine ({matdim}D) does not match target shape ({ndims}D).')
    validate_affine_shape(matrix.shape)

    # list of volume ndgrid
    # N-long list, each entry of shape
    mesh = ne.utils.volshape_to_meshgrid(shape, indexing=indexing)
    mesh = [tf.cast(f, 'float32') for f in mesh]

    if shift_center:
        mesh = [mesh[f] - (shape[f] - 1) / 2 for f in range(len(shape))]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [ne.utils.flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels

    # compute locations
    loc_matrix = tf.matmul(matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = tf.transpose(loc_matrix[:ndims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, list(shape) + [ndims])  # *shape x N

    # get shifts and return
    return loc - tf.stack(mesh, axis=ndims)

def validate_affine_shape(shape):
    """
    Validates whether the given input shape represents a valid affine matrix.
    Throws error if the shape is valid.
    Parameters:
        shape: List of integers of the form [..., N, N+1].
    """
    ndim = shape[-1] - 1
    actual = tuple(shape[-2:])
    if ndim not in (2, 3) or actual != (ndim, ndim + 1):
        raise ValueError(f'Affine matrix must be of shape (2, 3) or (3, 4), got {actual}.')

def is_affine_shape(shape):
    """
    Determins whether the given shape (single-batch) represents an
    affine matrix.
    Parameters:
        shape:  List of integers of the form [N, N+1], assuming an affine.
    """
    if len(shape) == 2 and shape[-1] != 1:
        validate_affine_shape(shape)
        return True
    return False

def transform(vol, loc_shift, interp_method='linear', indexing='ij', fill_value=None):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow
    Essentially interpolates volume vol at locations determined by loc_shift.
    This is a spatial transform in the sense that at location [x] we now have the data from,
    [x + shift] so we've moved data.
    Args:
        vol (Tensor): volume with size vol_shape or [*vol_shape, C]
            where C is the number of channels
        loc_shift: shift volume [*new_vol_shape, D] or [*new_vol_shape, C, D]
            where C is the number of channels, and D is the dimentionality len(vol_shape)
            If loc_shift is [*new_vol_shape, D], it applies to all channels of vol
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.
    Return:
        new interpolated volumes in the same size as loc_shift[0]
    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """
    import neurite as ne
    # parse shapes.
    # location volshape, including channels if available
    loc_volshape = loc_shift.shape[:-1]
    if isinstance(loc_volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        loc_volshape = loc_volshape.as_list()

    # volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])

    # location should be mesh and delta
    mesh = ne.utils.volshape_to_meshgrid(loc_volshape, indexing=indexing)  # volume mesh
    loc = [tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)]

    # if channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(tf.cast(mesh[-1], 'float32'))

    # test single
    return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)

