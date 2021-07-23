import os
import io
import logging
import tensorflow
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from src.visualization.Visualize import plot_3d_vol, plot_4d_vol
from src.visualization.Visualize import show_slice_transparent as show_slice
from src.utils.Utils_io import ensure_dir

from src.data.Preprocess import normalise_image, transform_to_binary_mask
from src.visualization.Visualize import show_2D_or_3D


def get_callbacks(config=None, batch_generator=None, validation_generator=None, metrics=None):
    """
    :param config:
    :param validation_generator:
    :param batch_generator:
    :return: list of callbacks for keras fit_generator
    """

    if config is None:
        config = {}
    callbacks = []
    ensure_dir(config['MODEL_PATH'])

    # add these callbacks for each training
    callbacks.append(
        ModelCheckpoint(os.path.join(config['MODEL_PATH'], 'model.h5'),
                        # could also be 'model.h5 to save only the weights
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=True,
                        monitor=config.get('SAVE_MODEL_FUNCTION', 'loss'),
                        mode=config.get('SAVE_MODEL_MODE', 'min'),
                        save_freq='epoch'))

    callbacks.append(
        tensorflow.keras.callbacks.ReduceLROnPlateau(monitor=config.get('MONITOR_FUNCTION', 'loss'),
                                                     factor=config.get('DECAY_FACTOR', 0.5),
                                                     patience=config.get('REDUCE_LR_ON_PLATEAU_PATIENCE', 5),
                                                     verbose=1,
                                                     cooldown=2,
                                                     mode=config.get('MONITOR_MODE', 'auto'),
                                                     min_lr=config.get('MIN_LR', 1e-12)))

    callbacks.append(
        LRTensorBoard(log_dir=config.get('TENSORBOARD_PATH', 'temp/tf_log'),
                      histogram_freq=0,
                      write_graph=False,
                      write_images=False,
                      update_freq='epoch',
                      profile_batch=0,
                      embeddings_freq=0))

    # if we provide a batch_generator we assume to that we want to observe the trainings progress
    # by intermediate predictions saved ad png or into the tensorboard
    if batch_generator:

        # depending on the task we might have different interfaces and imge, flowfield, onehot vectors etc.
        # standard behaviour, if no flags are given: dont add them!
        if config.get('SAVE_LEARNING_PROGRESS_AS_PNG', False):
            callbacks.append(
                ImageSaver(image_dir='reports/figures/{}/'.format(config.get('EXPERIMENT'), 'temp'),
                           image_freq=config.get('SAVE_LEARNING_PROGRESS_FREQUENCY', 2),
                           feed_inputs_4_display=feed_inputs_4_tensorboard(config, batch_generator,
                                                                           validation_generator),
                           flow=config.get('FLOW', False)))
        if config.get('SAVE_LEARNING_PROGRESS_AS_TF', False):
            callbacks.append(
                PhaseRegressionCallback(log_dir=config['TENSORBOARD_PATH'],
                                        image_freq=config.get('SAVE_LEARNING_PROGRESS_FREQUENCY', 2),
                                        feed_inputs_4_display=feed_inputs_4_tensorboard(config, batch_generator,
                                                                                        validation_generator),
                                        ))
        if config.get('SAVE_LEARNING_P2P_TF'):
            callbacks.append(
                WindowMotionCallback(log_dir=config['TENSORBOARD_PATH'],
                                     image_freq=config.get('SAVE_LEARNING_PROGRESS_FREQUENCY', 2),
                                     feed_inputs_4_display=feed_inputs_4_tensorboard(config, batch_generator,
                                                                                     validation_generator),
                                     take_t_elem=config.get('INPUT_T_ELEM', 0)
                                     ))

    if config.get('POLY_LR_DECAY', False):
        callbacks.append(
            LearningRateScheduler(schedule=
                                  PolynomialDecay(maxEpochs=config.get('EPOCHS', 100),
                                                  initAlpha=config.get('LEARNING_RATE', 1e-4, ),
                                                  power=2),
                                  verbose=1)
        )

    if metrics:  # optimizer will be changed to SGD, if adam does not improve any more
        # changer will call this method without metrics to avoid recursive learning
        logging.info('optimizer will be changed to SGD after adam does not improve any more')
        # idea based on: https://arxiv.org/pdf/1712.07628.pdf
        callbacks.append(
            OptimizerChanger(on_train_end=finetune_with_SGD,
                             train_generator=batch_generator,
                             val_generator=validation_generator,
                             config=config,
                             metrics=metrics,
                             patience=15,
                             verbose=1,
                             monitor=config.get('MONITOR_FUNCTION', 'loss'),
                             mode=config.get('MONITOR_MODE', 'min')
                             )
        )
    else:  # no metrics given, use early stopping callback to stop the training after 20 epochs
        callbacks.append(
            EarlyStopping(patience=config.get('EARLY_STOPPING_PATIENCE', 10),
                          verbose=1,
                          monitor=config.get('MONITOR_FUNCTION', 'loss'),
                          mode=config.get('MONITOR_MODE', 'min'))
        )

    return callbacks


def feed_inputs_4_tensorboard(config, batch_generator=None, validation_generator=None, samples=4):
    """
    Helper method it creates the data for the CustomImageWriter Callback
    Returns some sample images for visualisation in Tensorboard
    :param config: batchgenerator params
    :param batch_generator: could be provided, should inherit from keras.sequences
    :param validation_generator: could be provided, should inherit from keras.sequences
    :param samples: define the examples per generator and area
    :return: feed_inputs_4_display: dict {'train':(x_tensor,y_tensor), 'val' : (x_tensor. y_tensor), ...}
    """

    feed = {}
    # training config includes the generator args
    generator_args = config

    samples = min(config['BATCHSIZE'], samples)
    # use the batch- and validation-generator for the feeds
    if batch_generator is not None:
        x_t, y_t = batch_generator.__getitem__(0)

        if y_t is not None:
            x_t, y_t = x_t[0:samples], y_t[0:samples]
        else:
            x_t, y_t = x_t[0:samples], []

        feed['gen_train'] = (x_t, y_t)

    if validation_generator is not None:
        x_v, y_v = validation_generator.__getitem__(0)
        x_v, y_v = x_v[0:samples], y_v[0:samples]

        feed['gen_val'] = (x_v, y_v)

    logging.info('feed 4 Tensorboard is ready')
    return feed


class StepDecay:
    def __init__(self, init_alpha=0.01, factor=0.25, drop_every=10):
        self.init_alpha = init_alpha
        self.factor = factor
        self.drop_every = drop_every

    def __call__(self, epoch):
        exp = np.floor((1 + epoch) / self.drop_every)
        alpha = self.init_alpha * (self.factor ** exp)

        return float(alpha)


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


class TrainValTensorBoard(TensorBoard):

    def __init__(self, log_dir='./logs', **kwargs):

        """
        This Callback is neccesary to plot train and validation score in one subdirectory 
        :param log_dir: tensorboard summary folder
        :param kwargs: tensorboard config arguments
        """

        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        ensure_dir(training_log_dir)
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
        ensure_dir(self.val_log_dir)

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tensorflow.summary.create_file_writer(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        if epoch % 1 == 0:
            logs = logs or {}
            val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
            for name, value in val_logs.items():
                summary = tensorflow.summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, epoch)
            self.val_writer.flush()

            # Pass the remaining logs to `TensorBoard.on_epoch_end`
            logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
            super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


from tensorflow.keras.callbacks import LearningRateScheduler


class PolynomialDecay:
    def __init__(self, maxEpochs=100, initAlpha=0.01, power=0.25):
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power

    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay
        tensorflow.summary.scalar('learning rate', data=alpha, step=epoch)
        # return the new learning rate
        return float(alpha)


class OptimizerChanger(EarlyStopping):
    """
    Callback to switch the optimizer instead of early stopping the training
    """

    def __init__(self, on_train_end, train_generator, val_generator, config, metrics, **kwargs):
        self.do_on_train_end = on_train_end
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.config = config
        self.metrics = metrics
        self.current_epoch = 0
        super(OptimizerChanger, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        """
        remember the last epoch for the ontrainend callback
        :param epoch:
        :param logs:
        :return:
        """
        super(OptimizerChanger, self).on_epoch_end(epoch, logs)
        self.current_epoch = epoch

    def on_train_end(self, logs=None):
        """
        eventhandler from callbacks overwritten
        :param logs:
        :return:
        """

        super(OptimizerChanger, self).on_train_end(logs)
        self.do_on_train_end(self.config, self.train_generator, self.val_generator, self.model, self.metrics,
                             self.current_epoch)


def finetune_with_SGD(config, train_g, val_g, model, metrics, epoch_init):
    """
    injection funtion to finetune a trained model with the SGD optimizer
    :param config:
    :param train_g:
    :param val_g:
    :param model:
    :param metrics:
    :return:
    """
    import tensorflow as tf
    loss_f = config.get('LOSS_FUNCTION', tf.keras.metrics.binary_crossentropy)
    # loss_f = dice_coef_labels_loss
    lr = config.get('LEARNING_RATE', 0.001)
    # opt = tf.keras.optimizers.Adam(lr=lr)
    opt = tf.keras.optimizers.SGD(name='SGD')
    model.compile(optimizer=opt, loss=loss_f, metrics=metrics)
    model.fit(
        x=train_g,
        epochs=config['EPOCHS'],
        callbacks=get_callbacks(config, train_g, val_g),
        steps_per_epoch=len(train_g),
        validation_data=val_g,
        max_queue_size=20,
        initial_epoch=epoch_init,
        workers=0,
        verbose=1)


class SGDRScheduler(tf.keras.callbacks.Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.

    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.

    # References
        Original paper: http://arxiv.org/abs/1608.03983
    '''

    def __init__(self,
                 min_lr,
                 max_lr,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        self.steps_per_epoch = self.params['steps'] if self.params['steps'] is not None else round(
            self.params['samples'] / self.params['batch_size'])
        logs = logs or {}
        tf.keras.backend.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(tf.keras.backend.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        tf.keras.backend.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)


class CustomImageWritertf2(Callback):

    # Keras Callback for training progress visualisation in the Tensorboard
    # Creates a new summary file
    # Usage:
    # custom_image_writer = CustomImageWriter(experiment_path, 10, create_feeds_for_tensorboard(batch_generator, val_generator)
    # model.fit_generator(batch_generator, val_generator, *args, callbacks=[custom_image_writer, ...]
    # original code from:
    # https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1

    def __init__(self, log_dir='./logs/tmp/', image_freq=10, feed_inputs_4_display=None, flow=False, dpi=200,
                 f_size=(5, 5), interpol='bilinear', force_plot_first_n_epochs=1):

        """
        This callback gets a dict with key: x,y entries
        When the on_epoch_end callback is invoked this callback predicts the current output for all xs
        Afterwards it writes the image, gt and prediction into a summary file to make the learning visually in the Tensorboard
        :param log_dir: String, path - folder for the tensorboard summary file Imagewriter will create a subdir "images" for the imagesummary file
        :param image_freq: int - run this callback every n epoch to save disk space and increase speed
        :param feed_inputs_4_display: dict {'train':(x_tensor,y_tensor), 'val' : (x_tensor. y_tensor)}
        x and ys to predict and visualise + key for summary description
        x_tensor and y_tensor have the shape n, x, y, 1 or classes for y, they are grouped by a key, eg. 'train', 'val'
        """

        super(CustomImageWritertf2, self).__init__()
        self.freq = image_freq
        self.f_size = f_size
        self.dpi = dpi
        self.interpol = interpol
        self.flow = flow
        self.e = 0
        self.every_n_in_z = 5
        self.n_start_epochs = force_plot_first_n_epochs
        self.feed_inputs_4_display = feed_inputs_4_display
        log_dir = os.path.join(log_dir, 'images')  # create a subdir for the imagewriter summary file
        ensure_dir(log_dir)
        self.writer = tensorflow.summary.create_file_writer(log_dir)

    def custom_set_feed_input_to_display(self, feed_inputs_display):

        """
        sets the feeding data for TensorBoard visualization;
        :param feed_inputs_display: dict {'train':(x_tensor,y_tensor), 'val' : (x_tensor. y_tensor)}
        x and ys to predict and visualise + key for summary description
        x_tensor and y_tensor have the shape n, x, y, 1 or classes for y, they are grouped by a key, eg. 'train', 'val'
        :return: None
        """

        self.feed_inputs_display = feed_inputs_display

    def make_image(self, figure):

        """
        Create a tf.Summary.Image from an ndarray
        :param numpy_img: Greyscale image with shape (x, y, 1)
        :return:
        """
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
          returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tensorflow.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tensorflow.expand_dims(image, 0)
        return image

    def on_epoch_end(self, epoch, logs=None):

        """
        Keras will call this methods after each epoch on all Callbacks provided to the method fit or fit_generator
        A callback has access to its associated model through the class property self.model.
        :param epoch:
        :param logs:
        :return:
        """

        logs = logs or {}
        self.e += 1
        if self.e % self.freq == 0 or self.e < self.n_start_epochs:  # every n epoch (and the first 20 epochs), write pred in a TensorBorad summary file;
            summary_str = []

            # xs and ys have the shape n, x, y, 1, they are grouped by the key
            # xs will have the shape: (len(keys), samples, z, x, y, 1)
            # need to reshape with len(keys) x samples
            xs, ys = zip(*self.feed_inputs_4_display.values())
            keys = self.feed_inputs_4_display.keys()

            # reshape x to predict in one step
            x_ = np.stack(xs, axis=0)
            x_ = x_.reshape((x_.shape[0] * x_.shape[1], *x_.shape[2:]))
            predictions = self.model.predict(x_)

            # create one tensorboard entry per key in feed_inputs_display
            pred_i = 0
            with self.writer.as_default():
                for key, x, y in zip(keys, xs, ys):
                    # xs and ys have the shape n, x, y, 1, they are grouped by the key
                    # count the samples provided by each key to sort them
                    for i in range(x.shape[0]):
                        # pred has the same order as x and y but no grouping tag (e.g. 'train_generator')
                        # keep track of the matching
                        if len(x.shape) == 4:  # work with 2d data
                            pred = predictions[pred_i]
                            if not self.flow:
                                tensorflow.summary.image(name='plot/{}/{}/_prediction'.format(key, i),
                                                         data=self.make_image(
                                                             show_slice(img=x[i], mask=pred, show=False,
                                                                        f_size=self.f_size, dpi=self.dpi,
                                                                        interpol=self.interpol)),
                                                         step=epoch)

                                tensorflow.summary.image(name='plot/{}/{}/_ground_truth'.format(key, i),
                                                         data=self.make_image(
                                                             show_slice(img=x[i], mask=y[i], show=False,
                                                                        f_size=self.f_size, dpi=self.dpi,
                                                                        interpol=self.interpol)),
                                                         step=0)
                            pred_i += 1
                        if len(x.shape) == 5:  # work with 3d data

                            for z in range(x.shape[1]):  # for each slice in this 3d image
                                # plot only every n slice (compute time)
                                if z % self.every_n_in_z == 0:
                                    pred = predictions[pred_i][z]
                                    if not self.flow:
                                        tensorflow.summary.image(name='plot/{}/{}/_prediction/{}'.format(key, i, z),
                                                                 data=self.make_image(
                                                                     show_slice(img=x[i][z], mask=pred, show=False,
                                                                                f_size=self.f_size, dpi=self.dpi,
                                                                                interpol=self.interpol)),
                                                                 step=epoch)

                                        tensorflow.summary.image(name='plot/{}/{}/ground_truth{}'.format(key, i, z),
                                                                 data=self.make_image(
                                                                     show_slice(img=x[i][z], mask=y[i][z], show=False,
                                                                                f_size=self.f_size, dpi=self.dpi,
                                                                                interpol=self.interpol)),
                                                                 step=0)

                                    else:  # work with
                                        tensorflow.summary.image(name='plot/{}/{}/_prediction/{}'.format(key, i, z),
                                                                 data=self.make_image(
                                                                     normalise_image(pred)),
                                                                 step=epoch)

                                        tensorflow.summary.image(name='plot/{}/{}/ground_truth{}'.format(key, i, z),
                                                                 data=self.make_image(
                                                                     normalise_image(x[i][z])),
                                                                 step=0)

                            pred_i += 1
            # del xs, ys, pred

            # self.writer.add_summary(tf.Summary(value=summary_str), global_step=self.e)


class WindowMotionCallback(Callback):

    # Keras Callback for training progress visualisation in the Tensorboard
    # Creates a new summary file
    # Usage:
    # custom_image_writer = CustomImageWriter(experiment_path, 10, create_feeds_for_tensorboard(batch_generator, val_generator)
    # model.fit_generator(batch_generator, val_generator, *args, callbacks=[custom_image_writer, ...]
    # original code from:
    # https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1

    def __init__(self, log_dir='./logs/tmp/', image_freq=10, feed_inputs_4_display=None, dpi=200, f_size=(5, 5),
                 interpol='bilinear', force_plot_first_n_epochs=20, take_t_elem=0):

        """
        This callback gets a dict with key: x,y entries
        When the on_epoch_end callback is invoked this callback predicts the current output for all xs
        Afterwards it writes the image, gt and prediction into a summary file to make the learning visually in the Tensorboard
        :param log_dir: String, path - folder for the tensorboard summary file Imagewriter will create a subdir "images" for the imagesummary file
        :param image_freq: int - run this callback every n epoch to save disk space and increase speed
        :param feed_inputs_4_display: dict {'train':(x_tensor,y_tensor), 'val' : (x_tensor. y_tensor)}
        x and ys to predict and visualise + key for summary description
        x_tensor and y_tensor have the shape n, x, y, 1 or classes for y, they are grouped by a key, eg. 'train', 'val'
        """

        super(WindowMotionCallback, self).__init__()
        self.freq = image_freq
        self.f_size = f_size
        self.dpi = dpi
        self.interpol = interpol
        self.e = 0
        self.every_n_in_z = 5
        self.n_start_epochs = force_plot_first_n_epochs
        self.feed_inputs_4_display = feed_inputs_4_display
        log_dir = os.path.join(log_dir, 'images')  # create a subdir for the imagewriter summary file
        ensure_dir(log_dir)
        self.writer = tensorflow.summary.create_file_writer(log_dir)
        self.xs, self.ys = zip(*self.feed_inputs_4_display.values())
        self.keys = self.feed_inputs_4_display.keys()
        self.take_t_elem = take_t_elem

    def custom_set_feed_input_to_display(self, feed_inputs_display):

        """
        sets the feeding data for TensorBoard visualization;
        :param feed_inputs_display: dict {'train':(x_tensor,y_tensor), 'val' : (x_tensor. y_tensor)}
        x and ys to predict and visualise + key for summary description
        x_tensor and y_tensor have the shape n, x, y, 1 or classes for y, they are grouped by a key, eg. 'train', 'val'
        :return: None
        """

        self.feed_inputs_display = feed_inputs_display

    def make_image(self, figure):

        """
        Create a tf.Summary.Image from an ndarray
        Converts the matplotlib plot specified by 'figure' to a PNG image and
          returns it. The supplied figure is closed and inaccessible after this call.
        :param numpy_img: Greyscale image with shape (x, y, 1)
        :return:
        """
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tensorflow.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tensorflow.expand_dims(image, 0)
        return image

    def on_train_begin(self, logs=None):
        # Call the image writer callback once before training
        self.on_epoch_end(epoch=0, logs=logs)

    def on_epoch_end(self, epoch, logs=None):

        """
        Keras will call this methods after each epoch on all Callbacks provided to the method fit or fit_generator
        A callback has access to its associated model through the class property self.model.
        :param epoch:
        :param logs:
        :return:
        """

        logs = logs or {}
        self.e += 1
        if self.e % self.freq == 0 or self.e < self.n_start_epochs:  # every n epoch (and the first 20 epochs), write pred in a TensorBorad summary file;
            summary_str = []

            # xs and ys have the shape n, x, y, 1, they are grouped by the key
            # xs will have the shape: (len(keys), samples, z, x, y, 1)

            # create one tensorboard entry per key in feed_inputs_display
            pred_i = 0
            with self.writer.as_default():

                for key, x, y in zip(self.keys, self.xs, self.ys):

                    pred_ = self.model.predict(x)
                    if len(pred_) == 2:
                        movings, vects = pred_
                    else:
                        movings, moving_m, vects = pred_
                    # logging.info(predictions.shape)
                    # xs and ys have the shape n, x, y, 1, they are grouped by the key
                    # count the samples provided by each key to sort them

                    phases = ['ED#', 'MS#', 'ES#', 'PF#', 'MD#']
                    b = 0
                    # one plot per phase
                    for p in range(len(phases)):

                        # Define the plot grid size
                        nrows = 3
                        ncols = 7
                        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7))
                        # Slice the
                        first_vol, second_vol = x[b][0][p], y[b][0][p]
                        first_m, second_m = x[1][0][p], y[1][0][p]
                        if first_vol.shape[-1] == 3:
                            first_vol = first_vol[..., self.take_t_elem][..., np.newaxis]
                        if first_m.shape[-1] == 3:
                            first_m = first_m[..., self.take_t_elem]
                        moved, moved_m, vect = movings[b][p], moving_m[b][p], vects[b][p]
                        spatial_slices = first_vol.shape[0]
                        # pick one upper, middle and lower slice as example
                        picks = (np.array([0.6, 0.4, 0.2]) * spatial_slices).astype(int)
                        y_label = ['Basal', 'Mid', 'Apex']
                        from tensorflow.keras.metrics import mse
                        mse_1 = np.mean((first_vol - second_vol) ** 2)
                        mse_2 = np.mean((moved - second_vol) ** 2)
                        col_titles = ['t1', 't2', 't1 moved', 'vect', 'magn', 't1-t2 \n {:6.4f}'.format(mse_1),
                                      'moved-t2 \n {:6.4f}'.format(mse_2)]
                        vmax = 1
                        for i, z in enumerate(picks):
                            j = 0
                            # t0
                            axes[i, j].imshow(normalise_image(first_vol[z]), 'gray')
                            axes[i, j].imshow(first_m[z], alpha=0.6)
                            axes[i, j].set_ylabel(y_label[i], rotation=90, size='medium')
                            axes[i, j].set_xticks([])
                            axes[i, j].set_yticks([])
                            j = j + 1
                            # t1
                            axes[i, j].imshow(second_vol[z], 'gray')
                            axes[i, j].imshow(second_m[z], alpha=0.6)
                            axes[i, j].set_xticks([])
                            axes[i, j].set_yticks([])
                            j = j + 1
                            # moved t0
                            axes[i, j].imshow(moved[z], 'gray')
                            axes[i, j].imshow(moved_m[z], alpha=0.6)
                            axes[i, j].set_xticks([])
                            axes[i, j].set_yticks([])
                            j = j + 1
                            # vect, abs & min/max normalized
                            temp = np.absolute(vect[z])
                            axes[i, j].imshow(first_vol[z], 'gray', vmin=0, vmax=0.8)
                            axes[i, j].imshow(normalise_image(temp), alpha=0.8)
                            axes[i, j].set_xticks([])
                            axes[i, j].set_yticks([]);
                            j = j + 1

                            # magnitude
                            temp = normalise_image(np.sqrt(
                                np.square(vect[z][..., 0]) + np.square(vect[z][..., 1]) + np.square(vect[z][..., 2])))
                            axes[i, j].imshow(first_vol[z], 'gray', vmin=0, vmax=.8)
                            axes[i, j].imshow(temp, cmap='seismic', alpha=0.8)

                            axes[i, j].set_xticks([])
                            axes[i, j].set_yticks([]);
                            j = j + 1
                            # diff t0 - t1
                            axes[i, j].imshow(first_vol[z], 'gray', vmin=0, vmax=.8)
                            axes[i, j].imshow(np.abs(first_vol[z] - second_vol[z]), cmap='seismic',
                                              interpolation='none')
                            axes[i, j].set_xticks([])
                            axes[i, j].set_yticks([]);
                            j = j + 1
                            # diff moved - t1
                            axes[i, j].imshow(first_vol[z], 'gray', vmin=0, vmax=.8)
                            axes[i, j].imshow(np.abs(moved[z] - second_vol[z]), cmap='seismic', interpolation='none')
                            axes[i, j].set_xticks([])
                            axes[i, j].set_yticks([])
                        # set column names
                        for i in range(ncols):
                            axes[0, i].set_title(col_titles[i])
                        # set row names
                        #                        for j in range(nrows):
                        #                            axes[j,0].set_ylabel(y_label[j], rotation=90, size='medium')

                        fig.subplots_adjust(wspace=0.0, hspace=0.0)
                        # fig.tight_layout()
                        tensorflow.summary.image(name='plot/{}/batch_{}/{}/summary'.format(key, b, phases[p]),
                                                 data=self.make_image(fig),
                                                 step=epoch)


class PhaseRegressionCallback(Callback):

    # Keras Callback for training progress visualisation in the Tensorboard
    # Creates a new summary file
    # Usage:
    # custom_image_writer = CustomImageWriter(experiment_path, 10, create_feeds_for_tensorboard(batch_generator, val_generator)
    # model.fit_generator(batch_generator, val_generator, *args, callbacks=[custom_image_writer, ...]
    # original code from:
    # https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1

    def __init__(self, log_dir='./logs/tmp/', image_freq=10, feed_inputs_4_display=None, dpi=200, f_size=(5, 5),
                 interpol='bilinear', force_plot_first_n_epochs=5):

        """
        This callback gets a dict with key: x,y entries
        When the on_epoch_end callback is invoked this callback predicts the current output for all xs
        Afterwards it writes the image, gt and prediction into a summary file to make the learning visually in the Tensorboard
        :param log_dir: String, path - folder for the tensorboard summary file Imagewriter will create a subdir "images" for the imagesummary file
        :param image_freq: int - run this callback every n epoch to save disk space and increase speed
        :param feed_inputs_4_display: dict {'train':(x_tensor,y_tensor), 'val' : (x_tensor. y_tensor)}
        x and ys to predict and visualise + key for summary description
        x_tensor and y_tensor have the shape n, x, y, 1 or classes for y, they are grouped by a key, eg. 'train', 'val'
        """

        super(PhaseRegressionCallback, self).__init__()
        self.freq = image_freq
        self.f_size = f_size
        self.dpi = dpi
        self.interpol = interpol
        self.e = 0
        self.every_n_in_z = 5
        self.n_start_epochs = force_plot_first_n_epochs
        self.feed_inputs_4_display = feed_inputs_4_display
        log_dir = os.path.join(log_dir, 'images')  # create a subdir for the imagewriter summary file
        ensure_dir(log_dir)
        self.writer = tensorflow.summary.create_file_writer(log_dir)
        self.xs, self.ys = zip(*self.feed_inputs_4_display.values())
        self.keys = self.feed_inputs_4_display.keys()

        # reshape x to predict in one step
        x_ = np.stack(self.xs, axis=0)
        self.x_ = x_.reshape((x_.shape[0] * x_.shape[1], *x_.shape[2:]))

    def custom_set_feed_input_to_display(self, feed_inputs_display):

        """
        sets the feeding data for TensorBoard visualization;
        :param feed_inputs_display: dict {'train':(x_tensor,y_tensor), 'val' : (x_tensor. y_tensor)}
        x and ys to predict and visualise + key for summary description
        x_tensor and y_tensor have the shape n, x, y, 1 or classes for y, they are grouped by a key, eg. 'train', 'val'
        :return: None
        """

        self.feed_inputs_display = feed_inputs_display

    def make_image(self, figure):

        """
        Create a tf.Summary.Image from an ndarray
        :param numpy_img: Greyscale image with shape (x, y, 1)
        :return:
        """
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
          returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tensorflow.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tensorflow.expand_dims(image, 0)
        return image

    def on_epoch_end(self, epoch, logs=None):

        """
        Keras will call this methods after each epoch on all Callbacks provided to the method fit or fit_generator
        A callback has access to its associated model through the class property self.model.
        :param epoch:
        :param logs:
        :return:
        """

        logs = logs or {}
        self.e += 1
        if self.e % self.freq == 0 or self.e < self.n_start_epochs:  # every n epoch (and the first 20 epochs), write pred in a TensorBorad summary file;
            summary_str = []

            # xs and ys have the shape n, x, y, 1, they are grouped by the key
            # xs will have the shape: (len(keys), samples, z, x, y, 1)
            from src.visualization.Visualize import show_phases

            # create one tensorboard entry per key in feed_inputs_display
            pred_i = 0
            with self.writer.as_default():

                for key, x, y in zip(self.keys, self.xs, self.ys):
                    predictions = self.model.predict(x)
                    if len(predictions) == 3:  # multi-output-model
                        predictions, y = predictions[0], y[0]
                    # logging.info(predictions.shape)
                    # xs and ys have the shape n, x, y, 1, they are grouped by the key
                    # count the samples provided by each key to sort them
                    tensorflow.summary.image(name='plot/{}/_pred'.format(key, pred_i),
                                             data=self.make_image(show_phases(y, predictions)),
                                             step=epoch)


class ImageSaver(Callback):

    # Keras Callback for training progress visualisation in the Tensorboard
    # Creates a new summary file
    # Usage:
    # custom_image_writer = CustomImageWriter(experiment_path, 10, create_feeds_for_tensorboard(batch_generator, val_generator)
    # model.fit_generator(batch_generator, val_generator, *args, callbacks=[custom_image_writer, ...]
    # original code from:
    # https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1

    def __init__(self, image_dir='./reports/tmp/figures/', image_freq=2, feed_inputs_4_display=None, flow=False,
                 dpi=200, f_size=(5, 5), interpol='bilinear'):

        """
        This callback gets a dict with key: x,y entries
        When the on_epoch_end callback is invoked this callback predicts the current output for all xs
        Afterwards it writes the image, gt and prediction into a summary file to show the learning progress in the Tensorboard
        :param log_dir: String, path - folder for the tensorboard summary file Imagewriter will create a subdir "images" for the imagesummary file
        :param image_freq: int - run this callback every n epoch to save disk space and increase speed
        :param feed_inputs_4_display: dict {'train':(x_tensor,y_tensor), 'val' : (x_tensor. y_tensor)}
        x and ys to predict and visualise + key for summary description
        x_tensor and y_tensor have the shape n, x, y, 1 or classes for y, they are grouped by a key, eg. 'train', 'val'
        """

        super(self.__class__, self).__init__()
        self.freq = image_freq
        self.flow = flow
        self.f_size = f_size
        self.dpi = dpi
        self.interpol = interpol
        self.e = 0
        self.n_start_epochs = 20
        self.feed_inputs_4_display = feed_inputs_4_display
        self.image_dir = os.path.join(image_dir,
                                      'trainings_progress')  # create a subdir for the imagewriter summary file
        ensure_dir(self.image_dir)

    def custom_set_feed_input_to_display(self, feed_inputs_display):

        """
        sets the feeding data for TensorBoard visualization;
        :param feed_inputs_display: dict {'train':(x_tensor,y_tensor), 'val' : (x_tensor. y_tensor)}
        x and ys to predict and visualise + key for summary description
        x_tensor and y_tensor have the shape n, x, y, 1 or classes for y, they are grouped by a key, eg. 'train', 'val'
        :return: None
        """

        self.feed_inputs_display = feed_inputs_display

    def save_image(self, figure, fname, dpi=200):

        """
        save figure as png with given resolution
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        return it. The supplied figure is closed and inaccessible after this call.
        :param numpy_img: Greyscale image with shape (x, y, 1)
        :return:
        """
        # Save the plot as PNG in memory.
        fname = os.path.join(self.image_dir, fname)

        plt.savefig(fname, format='png', dpi=dpi)
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        return True

    def on_epoch_end(self, epoch, logs=None):

        """
        Keras will call this methods after each epoch on all Callbacks provided to the method fit or fit_generator
        A callback has access to its associated model through the class property self.model.
        :param epoch:
        :param logs:
        :return:
        """

        logs = logs or {}
        self.e += 1
        if self.e % self.freq == 0 or self.e < self.n_start_epochs:  # every n epoch (and the first 20 epochs), write pred in a TensorBorad summary file;
            summary_str = []

            # xs and ys have the shape n, x, y, 1, they are grouped by the key
            # xs will have the shape: (len(keys), samples, z, x, y, 1)
            # need to reshape with len(keys) x samples
            xs, ys = zip(*self.feed_inputs_4_display.values())
            keys = self.feed_inputs_4_display.keys()

            # reshape x to predict in one step
            x_ = np.stack(xs, axis=0)
            x_ = x_.reshape((x_.shape[0] * x_.shape[1], *x_.shape[2:]))
            predictions = self.model.predict(x_)

            # create one tensorboard entry per key in feed_inputs_display
            pred_i = 0

            for key, x, y in zip(keys, xs, ys):
                # xs and ys have the shape n, x, y, 1, they are grouped by the key
                # count the samples provided by each key to sort them
                for i in range(x.shape[0]):
                    # pred has the same order as x and y but no grouping tag (e.g. 'train_generator')
                    # keep track of the matching
                    if len(x.shape) == 4:  # work with 2d data
                        pred = predictions[pred_i]
                        if not self.flow:
                            self.save_image(show_slice(img=x[i], mask=pred, show=False, f_size=self.f_size),
                                            'pred_{}_img{}_epoch{}.png'.format(key, i, epoch), dpi=300)

                            if epoch == 1:  # save the gt image only once
                                self.save_image(show_slice(img=x[i], mask=y[i], show=False, f_size=self.f_size),
                                                'gt_{}_img{}_epoch{}.png'.format(key, i, epoch), dpi=self.dpi)

                        pred_i += 1
                    if len(x.shape) == 5:  # work with 3d data

                        for z in range(x.shape[1]):  # for each slice in this 3d image
                            pred = predictions[pred_i][z]
                            if not self.flow:

                                self.save_image(show_slice(img=x[i][z], mask=pred, show=False),
                                                'pred_{}_img{}_{}_epoch{}'.format(key, i, z, epoch), dpi=300)

                                if epoch == 1:
                                    self.save_image(show_slice(img=x[i][z], mask=y[i][z], show=False),
                                                    'gt_{}_img{}_{}_epoch{}'.format(key, i, z, epoch), dpi=300)


                            else:  # work with flowfields
                                raise NotImplemented('saving predicted flow fields is not supported so far')
                                tensorflow.summary.image(name='plot/{}/{}/_prediction/{}'.format(key, i, z),
                                                         data=self.make_image(
                                                             normalise_image(pred)),
                                                         step=epoch)

                                tensorflow.summary.image(name='plot/{}/{}/ground_truth{}'.format(key, i, z),
                                                         data=self.make_image(
                                                             normalise_image(x[i][z])),
                                                         step=0)

                        pred_i += 1
            # del xs, ys, pred

            # self.writer.add_summary(tf.Summary(value=summary_str), global_step=self.e)


class WeightsSaver(Callback):
    """
    write a json graph description at the beginning of each training,
    Could be used to save the weights as h5 file.
    """

    def __init__(self, model_path, model_freq):
        self.model_path = model_path
        self.N = model_freq
        self.epoch_w = 0
        ensure_dir(model_path)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_w += 1
        if self.epoch_w == 1:
            try:
                model_json = self.model.to_json()
                model_path = self.model_path
                ensure_dir(model_path)
                self.model.save(model_path)
                with open(os.path.join(model_path, 'model.json'), "w") as json_file:
                    json_file.write(model_json)
            except Exception as e:
                logging.error(str(e))
        if self.epoch_w % self.N == 0:  # to save continous files
            # Save the model
            try:
                tf.keras.models.save_model(self.model, filepath=self.model_path, overwrite=True,
                                           include_optimizer=False, save_format='tf')
            except Exception as e:
                model_path = self.model_path
                ensure_dir(model_path)
                self.model.save_weights(os.path.join(model_path, 'checkpoint.h5'))

            # serialize weights to HDF5, could be done with ModelCheckpoint Callback
            # name = 'weights_e-{0}_val_loss-{1}.h5'.format(self.epoch_w, str(logs['val_loss'])[:4])
            # self.model.save_weights(os.path.join(model_path, name))
            logging.info("Saved model weightsto disk: {}".format(self.model_path))
