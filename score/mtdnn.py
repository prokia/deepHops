# --*-- coding: utf-8 --*--

import collections
import logging
import time
from collections import Sequence

import tensorflow as tf
from deepchem.models import KerasModel
from deepchem.models.layers import SwitchedDropout
from tensorflow.keras.layers import Input, Dense, Reshape, Activation, Lambda
import deepchem as dc
import numpy as np

logger = logging.getLogger(__name__)


class Mtdnn(KerasModel):
    """A fully connected network for multitask regression.

    This class provides lots of options for customizing aspects of the model: the
    number and widths of layers, the activation functions, regularization methods,
    etc.

    It optionally can compose the model from pre-activation residual blocks, as
    described in https://arxiv.org/abs/1603.05027, rather than a simple stack of
    dense layers.  This often leads to easier training, especially when using a
    large number of layers.  Note that residual blocks can only be used when
    successive layers have the same width.  Wherever the layer width changes, a
    simple dense layer will be used even if residual=True.
    """

    def __init__(self,
                 n_tasks,
                 n_features,
                 layer_sizes=[1000],
                 weight_init_stddevs=0.02,
                 bias_init_consts=1.0,
                 weight_decay_penalty=0.0,
                 weight_decay_penalty_type="l2",
                 dropouts=0.5,
                 activation_fns=tf.nn.relu,
                 uncertainty=False,
                 residual=False,
                 **kwargs):
        """Create a MultitaskRegressor.

        In addition to the following arguments, this class also accepts all the keywork arguments
        from TensorGraph.

        Parameters
        ----------
        n_tasks: int
          number of tasks
        n_features: int
          number of features
        layer_sizes: list
          the size of each dense layer in the network.  The length of this list determines the number of layers.
        weight_init_stddevs: list or float
          the standard deviation of the distribution to use for weight initialization of each layer.  The length
          of this list should equal len(layer_sizes)+1.  The final element corresponds to the output layer.
          Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
        bias_init_consts: list or float
          the value to initialize the biases in each layer to.  The length of this list should equal len(layer_sizes)+1.
          The final element corresponds to the output layer.  Alternatively this may be a single value instead of a list,
          in which case the same value is used for every layer.
        weight_decay_penalty: float
          the magnitude of the weight decay penalty to use
        weight_decay_penalty_type: str
          the type of penalty to use for weight decay, either 'l1' or 'l2'
        dropouts: list or float
          the dropout probablity to use for each layer.  The length of this list should equal len(layer_sizes).
          Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
        activation_fns: list or object
          the Tensorflow activation function to apply to each layer.  The length of this list should equal
          len(layer_sizes).  Alternatively this may be a single value instead of a list, in which case the
          same value is used for every layer.
        uncertainty: bool
          if True, include extra outputs and loss terms to enable the uncertainty
          in outputs to be predicted
        residual: bool
          if True, the model will be composed of pre-activation residual blocks instead
          of a simple stack of dense layers.
        """
        self.n_tasks = n_tasks
        self.n_features = n_features
        n_layers = len(layer_sizes)
        if not isinstance(weight_init_stddevs, collections.Sequence):
            weight_init_stddevs = [weight_init_stddevs] * (n_layers + 1)
        if not isinstance(bias_init_consts, collections.Sequence):
            bias_init_consts = [bias_init_consts] * (n_layers + 1)
        if not isinstance(dropouts, collections.Sequence):
            dropouts = [dropouts] * n_layers
        if not isinstance(activation_fns, collections.Sequence):
            activation_fns = [activation_fns] * n_layers
        if weight_decay_penalty != 0.0:
            if weight_decay_penalty_type == 'l1':
                regularizer = tf.keras.regularizers.l1(weight_decay_penalty)
            else:
                regularizer = tf.keras.regularizers.l2(weight_decay_penalty)
        else:
            regularizer = None
        if uncertainty:
            if any(d == 0.0 for d in dropouts):
                raise ValueError(
                    'Dropout must be included in every layer to predict uncertainty')

        # Add the input features.

        mol_features = Input(shape=(n_features,))
        dropout_switch = Input(shape=tuple())
        prev_layer = mol_features
        prev_size = n_features
        next_activation = None

        # Add the dense layers

        for size, weight_stddev, bias_const, dropout, activation_fn in zip(
                layer_sizes, weight_init_stddevs, bias_init_consts, dropouts,
                activation_fns):
            layer = prev_layer
            if next_activation is not None:
                layer = Activation(next_activation)(layer)
            layer = Dense(
                size,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=weight_stddev),
                bias_initializer=tf.constant_initializer(value=bias_const),
                kernel_regularizer=regularizer)(layer)
            if dropout > 0.0:
                layer = SwitchedDropout(rate=dropout)([layer, dropout_switch])
            if residual and prev_size == size:
                prev_layer = Lambda(lambda x: x[0] + x[1])([prev_layer, layer])
            else:
                prev_layer = layer
            prev_size = size
            next_activation = activation_fn
        if next_activation is not None:
            prev_layer = Activation(activation_fn)(prev_layer)
        self.neural_fingerprint = prev_layer
        output = Reshape((n_tasks, 1))(Dense(
            n_tasks,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=weight_init_stddevs[-1]),
            bias_initializer=tf.constant_initializer(
                value=bias_init_consts[-1]))(prev_layer))
        if uncertainty:
            log_var = Reshape((n_tasks, 1))(Dense(
                n_tasks,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                    stddev=weight_init_stddevs[-1]),
                bias_initializer=tf.constant_initializer(value=0.0))(prev_layer))
            var = Activation(tf.exp)(log_var)
            outputs = [output, var, output, log_var]
            output_types = ['prediction', 'variance', 'loss', 'loss']

            def loss(outputs, labels, weights):
                diff = labels[0] - outputs[0]
                return tf.reduce_mean(diff * diff / tf.exp(outputs[1]) + outputs[1])
        else:
            outputs = [output]
            output_types = ['prediction']
            loss = dc.models.losses.L2Loss()
        model = tf.keras.Model(
            inputs=[mol_features, dropout_switch], outputs=outputs)
        # self.early_stop_callback = kwargs.pop('early_stop_callback')

        super(Mtdnn, self).__init__(
            model, loss, output_types=output_types, **kwargs)

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        for epoch in range(epochs):
            for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
                    batch_size=self.batch_size,
                    deterministic=deterministic,
                    pad_batches=pad_batches):
                if mode == 'predict':
                    dropout = np.array(0.0)
                else:
                    dropout = np.array(1.0)
                yield ([X_b, dropout], [y_b], [w_b])

    def fit_generator(self,
                      generator,
                      max_checkpoints_to_keep=5,
                      checkpoint_interval=1000,
                      restore=False,
                      variables=None,
                      loss=None,
                      callbacks=[]):
        """Train this model on data from a generator.

        Parameters
        ----------
        generator: generator
          this should generate batches, each represented as a tuple of the form
          (inputs, labels, weights).
        max_checkpoints_to_keep: int
          the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        checkpoint_interval: int
          the frequency at which to write checkpoints, measured in training steps.
          Set this to 0 to disable automatic checkpointing.
        restore: bool
          if True, restore the model from the most recent checkpoint and continue training
          from there.  If False, retrain the model from scratch.
        variables: list of tf.Variable
          the variables to train.  If None (the default), all trainable variables in
          the model are used.
        loss: function
          a function of the form f(outputs, labels, weights) that computes the loss
          for each batch.  If None (the default), the model's standard loss function
          is used.
        callbacks: function or list of functions
          one or more functions of the form f(model, step) that will be invoked after
          every step.  This can be used to perform validation, logging, etc.

        Returns
        -------
        the average loss over the most recent checkpoint interval
        """
        if not isinstance(callbacks, Sequence):
            callbacks = [callbacks]
        self._ensure_built()
        if checkpoint_interval > 0:
            manager = tf.train.CheckpointManager(self._checkpoint, self.model_dir,
                                                 max_checkpoints_to_keep)
        avg_loss = 0.0
        averaged_batches = 0
        train_op = None
        if loss is None:
            loss = self._loss_fn
        var_key = None
        if variables is not None:
            var_key = tuple(v.ref() for v in variables)

            # The optimizer creates internal variables the first time apply_gradients()
            # is called for a new set of variables.  If that happens inside a function
            # annotated with tf.function it throws an exception, so call it once here.

            zero_grads = [tf.zeros(v.shape) for v in variables]
            self._tf_optimizer.apply_gradients(zip(zero_grads, variables))
        if var_key not in self._gradient_fn_for_vars:
            self._gradient_fn_for_vars[var_key] = self._create_gradient_fn(variables)
        apply_gradient_for_batch = self._gradient_fn_for_vars[var_key]
        time1 = time.time()

        # Main training loop.

        for batch in generator:
            self._create_training_ops(batch)
            if restore:
                self.restore()
                restore = False
            inputs, labels, weights = self._prepare_batch(batch)

            # Execute the loss function, accumulating the gradients.

            if len(inputs) == 1:
                inputs = inputs[0]

            batch_loss = apply_gradient_for_batch(inputs, labels, weights, loss)
            current_step = self._global_step.numpy()

            avg_loss += batch_loss

            # Report progress and write checkpoints.
            averaged_batches += 1
            should_log = (current_step % self.tensorboard_log_frequency == 0)
            if should_log:
                avg_loss = float(avg_loss) / averaged_batches
                logger.info(
                    'Ending global_step %d: Average loss %g' % (current_step, avg_loss))
                avg_loss = 0.0
                averaged_batches = 0

            if checkpoint_interval > 0 and current_step % checkpoint_interval == checkpoint_interval - 1:
                manager.save()
            # callback的返回值决定是否early stop
            should_stop = [c(self, current_step, batch_loss) for c in callbacks]
            if self.tensorboard and should_log:
                with self._summary_writer.as_default():
                    tf.summary.scalar('loss', batch_loss, current_step)
            if any(should_stop):
                break

            # Report final results.
        if averaged_batches > 0:
            avg_loss = float(avg_loss) / averaged_batches
            logger.info(
                'Ending global_step %d: Average loss %g' % (current_step, avg_loss))

        if checkpoint_interval > 0:
            manager.save()

        time2 = time.time()
        logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
        return avg_loss
