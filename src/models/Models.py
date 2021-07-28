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

        unet = create_unet(config, single_model=False, networkname='3D-Unet')
        st_layer = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=True,
                                                 name='deformable_layer')

        gap = tensorflow.keras.layers.GlobalAveragePooling3D(name='GAP_3D_Layer')

        # unstack along the temporal axis
        # added shuffling, which avoids the model to be biased by the order
        # unstack along t, yielding a list of 3D volumes

        import random
        # unstack along t yielding a list of 3D volumes
        inputs_spatial = input_tensor
        inputs_spatial = tf.unstack(input_tensor, axis=1, name='split_into_3D_vols')
        # first tests without shuffling, later we can add the shuffling
        """indicies = list(tf.range(len(inputs_spatial)))
        zipped = list(zip(inputs_spatial, indicies))
        random.shuffle(zipped)
        inputs_spatial, indicies = zip(*zipped)"""
        # feed the T x 3D volumes into the spatial encoder (3D conv)
        pre_flows = [unet(vol) for vol in inputs_spatial]

        #pre_flows, _ = zip(*sorted(zip(pre_flows, indicies), key=lambda tup: tup[1]))
        flows= [Conv_layer(vol) for vol in pre_flows]

        transformed = [st_layer([input_vol, flow]) for input_vol, flow in
                       zip(inputs_spatial, flows)]
        inputs_spatial = flows[:]
        print('Flowfield shape: {}'.format(inputs_spatial[0].shape))

        # add the magnitude as fourth channel
        tensor_magnitude = [tf.norm(vol, ord='euclidean', axis=-1, keepdims=True, name='flow2norm') for vol in inputs_spatial]
        inputs_spatial = [tf.concat([vol, norm], axis=-1, name='extend_flow_with_norm') for vol,norm in zip(inputs_spatial, tensor_magnitude)]
        print('inkl norm shape: {}'.format(inputs_spatial[0].shape))
        # How to downscale the in-plane and spatial resolution?
        # 1st idea: apply conv layers with a stride
        # b, t, 16, 64, 64, 3/4
        # conv with: n times 4,4,4 filters, valid/no border padding and a striding of 4
        conv_1 = Conv(filters=16, kernel_size=4, padding='valid', strides=4,
                          kernel_initializer=kernel_init,
                      activation=activation,
                          name='downsample_1')
        inputs_spatial = [conv_1(vol) for vol in inputs_spatial]
        print('first conv shape: {}'.format(inputs_spatial[0].shape))
        #  b, t, 4, 16, 16, n
        # conv with: n times 4,4,4 filters, valid/no border padding and a striding of 4
        conv_2 = Conv(filters=32, kernel_size=4, padding='valid', strides=4,
                      kernel_initializer=kernel_init,
                      activation=activation,
                      name='downsample_2')
        inputs_spatial = [conv_2(vol) for vol in inputs_spatial]
        print('second conv shape: {}'.format(inputs_spatial[0].shape))
        # b, t, 1, 4, 4, n
        # conv with: n times 4,4,4 filters, valid/no border padding and a striding of 4
        conv_3 = Conv(filters=64, kernel_size=(1,4,4), padding='valid', strides=(1,4,4),
                      kernel_initializer=kernel_init,
                      activation=activation,
                      name='downsample_3')
        inputs_spatial = [conv_3(vol) for vol in inputs_spatial]
        print('third conv shape: {}'.format(inputs_spatial[0].shape))


        # b, t, 1, 1, 1, n
        # global average pooling for T x 3D vols
        #inputs_spatial = [gap(vol) for vol in inputs_spatial]
        # 2nd idea: GAP with/without pre-conv layer which extracts motion features into the channels

        # 3rd idea use the tft.pca module to transform the downstream.
        # This transform reduces the dimension of input vectors to output_dim in a way that retains the maximal variance

        encode_flow=False
        if encode_flow:

            enc = [spatial_encoder(vol)[0] for vol in inputs_spatial]
            print('shape after flow encoding: {}'.format(enc[0].shape))
            inputs_spatial = [gap(vol) for vol in enc]


        if pre_gap_conv:
            gap_conv = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=(1, 3, 3), padding='valid',
                                              activation=activation, kernel_initializer=kernel_init, name='pre-conv')

            inputs_spatial = [tf.keras.layers.BatchNormalization(name='final-BN')(elem) for elem in inputs_spatial]
            inputs_spatial = [tf.keras.layers.Dropout(rate=0.5, name='final-dropout')(elem) for elem in inputs_spatial]
            inputs_spatial = [gap_conv(vol) for vol in inputs_spatial]


        inputs_spatial = tf.concat(inputs_spatial, axis=1, name='merge_3D_into_4D')
        inputs_spatial = tf.keras.layers.Reshape(target_shape=(36,64))(inputs_spatial)

        inputs = inputs_spatial

        # 36, 256
        if add_bilstm:
            print('add a bilstm layer with: {} lstm units'.format(lstm_units))
            print('Shape before LSTM layers: {}'.format(inputs.shape))
            forward_layer = LSTM(lstm_units, return_sequences=True, name='forward_LSTM')
            backward_layer = LSTM(lstm_units, activation='tanh', return_sequences=True,
                                  go_backwards=True, name='backward_LSTM')  # maybe change to tanh
            inputs = Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=inputs.shape)(inputs)
            print('Shape after bilstm layer: {}'.format(inputs.shape))

        # 36,64
        print('Shape after Bi-LSTM layer')
        print(inputs.shape)

        # input (36,encoding) output (36,5)
        # either 36,256 --> from the temp encoder or
        # 36,64 --> 64 --> number of BI-LSTM units
        # activation to linear
        onehot = tf.keras.layers.Conv1D(filters=PHASES, kernel_size=1, strides=1, padding='same', activation=final_activation,
                                        name='pre_onehot')(inputs)
        flows = tf.stack(flows, axis=1, name='stacked_flows')
        transformed = tf.stack(transformed, axis=1, name='stacked_transformed')
        """if final_activation == 'relu':
            onehot = tf.keras.layers.ReLU()(onehot)
        elif final_activation == 'softmax':
            # axis -1 --> one class per timestep, as we repeat the phases its not possible to softmax the phase
            onehot = tf.keras.activations.softmax(onehot, axis=-1)
        elif final_activation == 'sigmoid':
            onehot = tf.keras.activations.sigmoid(onehot)
        else:
            logging.info('No final activation given! Please check the "FINAL_ACTIVATION" param!')"""

        # 36, 5
        print('Shape after final conv layer')
        print(onehot.shape)

        # add empty tensor with one-hot shape to align with gt
        zeros = tf.zeros_like(onehot, name='zero_padding')
        onehot = tf.stack([onehot, zeros], axis=1, name='extend_onehot_by_zeros')

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
                'transformed': 1,
                'flows': 0.001}
            #losses = [mse]

        print('added loss: {}'.format(loss))
        model = Model(inputs=[input_tensor], outputs=outputs, name=networkname)
        model.compile(
            optimizer=get_optimizer(config, networkname),
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







