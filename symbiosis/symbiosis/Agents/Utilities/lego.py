import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers
import tensorflow.compat.v1.keras.regularizers as regularizers
from typing import Callable, Tuple, Any, List

__all__ = ["NetBlocks"]

class NetBlocks:

      graph = tf.get_default_graph()

      @staticmethod
      def op(name, scope=None):
          if scope:
             return NetBlocks.graph.get_operation_by_name('{}/{}'.format(scope, name)).outputs[0]
          return NetBlocks.graph.get_operation_by_name('{}'.format(name)).outputs[0]

      @staticmethod
      def clip_grads_by_norm(gradients: tf.Tensor, clip_norm: float) -> None:
          for i, (grad, var) in enumerate(gradients):
              if grad is not None:
                 gradients[i] = (tf.clip_by_norm(grad, clip_norm), var)

      @staticmethod
      def placeholder(shape: Any, dtype: tf.DType = None, name: str = None) -> tf.Tensor:
          dtype = tf.float32 if not dtype else dtype
          if type(shape).__name__ == "str":
             return tf.placeholder(dtype=dtype, shape=[], name=name)
          return tf.placeholder(dtype=dtype, shape=[None] + list(shape), name=name)

      class layers:

            @staticmethod
            def Conv2D(filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int] = (1, 1),
                       activation: str = "relu", padding: str = "valid", batch_norm: bool = False, time_distributed: bool = False,
                       **optional_params) -> Callable:
                def _op(tensor: tf.Tensor) -> tf.Tensor:
                    if time_distributed:
                       tensor_out = layers.TimeDistributed(layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                                                         padding=padding, **optional_params))(tensor)
                    else:
                       tensor_out = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                                  **optional_params)(tensor)
                    if batch_norm:
                       if time_distributed:
                          tensor_out = layers.TimeDistributed(layers.BatchNomalization())(tensor_out)
                       else:
                          tensor_out = layers.BatchNormalization()(tensor_out)
                    if time_distributed:
                       tensor_out = layers.TimeDistributed(layers.Activation(activation))(tensor_out)
                    else:
                       tensor_out = layers.Activation(activation)(tensor_out)
                    return tensor_out
                return _op

            @staticmethod
            def Dense(units: int, activation: str = "relu", batch_norm: bool = False, time_distributed: bool = False, **optional_params) -> Callable:
                def _op(tensor: tf.Tensor) -> tf.Tensor:
                    if time_distributed:
                       tensor_out = layers.TimeDistributed(layers.Dense(units=units, kernel_initializer=tf.initializers.glorot_normal(),
                                                                        **optional_params))(tensor)
                    else:
                       tensor_out = layers.Dense(units=units, kernel_initializer=tf.initializers.glorot_normal(), **optional_params)(tensor)
                    if batch_norm:
                       if time_distributed:
                          tensor_out = layers.TimeDistributed(layers.BatchNomalization())(tensor_out)
                       else:
                          tensor_out = layers.BatchNormalization()(tensor_out)
                    if time_distributed:
                       tensor_out = layers.TimeDistributed(layers.Activation(activation))(tensor_out)
                    else:
                       tensor_out = layers.Activation(activation)(tensor_out)
                    return tensor_out
                return _op

            @staticmethod
            def LSTM(units: int) -> Callable:
                def _op(tensor: tf.Tensor) -> tf.Tensor:
                    if tf.test.is_gpu_available():
                       return layers.CuDNNLSTM(units=units)
                    return layers.LSTM(units=units)
                return _op

            @staticmethod
            def ResidualBlock(filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int] = (1, 1),
                              batch_norm: bool = False, time_distributed: bool = False) -> Callable:
                def _op(tensor: tf.Tensor) -> tf.Tensor:
                    tensor_out = Netblocks.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                                                         batch_norm=batch_norm, time_distributed=time_distributed)(tensor)
                    tensor_out = Netblocks.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same",
                                                         activation="linear", batch_norm=batch_norm,
                                                         time_distributed=time_distributed)(tensor_out)
                    skip_out = tf.add(tensor, tensor_out)
                    if time_distributed:
                       return layers.TimeDistributed(layers.Activation("relu"))(skip_out)
                    return layers.Activation("relu")(skip_out)
                return _op

      class NN:      

            @staticmethod
            def Conv2DNature(batch_norm: bool = False, time_distributed: bool = False) -> Callable:
                def _op(tensor: tf.Tensor) -> tf.Tensor:
                    conv1 = NetBlocks.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4),
                                             batch_norm=batch_norm, time_distributed=time_distributed)(tensor)
                    conv2 = NetBlocks.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                             batch_norm=batch_norm, time_distributed=time_distributed)(conv1)
                    conv3 = NetBlocks.layers.Conv2D(filters=64, kernel_size=(3, 3), batch_norm=batch_norm,
                                             time_distributed=time_distributed)(conv2)
                    flattened = layers.TimeDistributed(layers.Flatten())(conv3) if time_distributed else layers.Flatten()(conv3)
                    return NetBlocks.layers.Dense(units=256, batch_norm=batch_norm, time_distributed=time_distributed)(flattened)
                return _op

            @staticmethod
            def Conv2DNatureDueling(batch_norm: bool = False, time_distributed: bool = False) -> Callable:
                def _op(tensor: tf.Tensor) -> tf.Tensor:
                    conv1 = NetBlocks.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4),
                                             batch_norm=batch_norm, time_distributed=time_distributed)(tensor)
                    conv2 = NetBlocks.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                             batch_norm=batch_norm, time_distributed=time_distributed)(conv1)
                    conv3 = NetBlocks.layers.Conv2D(filters=64, kernel_size=(3, 3), batch_norm=batch_norm,
                                             time_distributed=time_distributed)(conv2)
                    flattened = layers.TimeDistributed(layers.Flatten())(conv3) if time_distributed else layers.Flatten()(conv3)
                    return [NetBlocks.layers.Dense(units=256, batch_norm=batch_norm, time_distributed=time_distributed)(flattened),
                            NetBlocks.layers.Dense(units=256, batch_norm=batch_norm, time_distributed=time_distributed)(flattened)]
                return _op

            @staticmethod
            def AGZChessNet(batch_norm: bool = False) -> Callable:
                def _op(tensor: tf.Tensor) -> List[tf.Tensor, tf.Tensor]:
                    tensor_out = NetBlocks.layers.Conv2D(filters=256, kernel_size=(5, 5), padding="same", batch_norm=batch_norm,
                                                         kernel_regularizer=regularizers.l2(1e-4))(tensor)
                    for _ in range(7):
                        tensor_out = NetBlocks.layers.ResidualBlock(filters=256, kernel_size=(3, 3), batch_norm=batch_norm)(tensor_out)
                    policy_out = NetBlocks.layers.Conv2D(filters=2, kernel_size=(1, 1), batch_norm=batch_norm,
                                                         kernel_regularizer=regularizers.l2(1e-4))(tensor_out)
                    policy_out = layers.Flatten()(policy_out)
                    value_out = NetBlocks.layers.Conv2D(filters=4, kernel_size=(1, 1), batch_norm=batch_norm,
                                                        kernel_regularizer=regularizers.l2(1e-4))(tensor_out)
                    value_out = layers.Flatten()(value_out)
                    value_out = NetBlocks.layers.Dense(units=256, kernel_regularizer=regularizers.l2(1e-4))(value_out)
                    value_out = NetBlocks.layers.Dense(units=1, activation="tanh", kernel_regularizer=regularizers.l2(1e-4))(value_out)
                    return policy_out, value_out
                return _op

      class losses:

            @staticmethod
            def huber(predicted: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
                errors = tf.subtract(predicted, labels)
                return errors, tf.where(tf.abs(errors) < 1.0, tf.square(errors) * 0.5, 1.0 * (tf.abs(errors) - 0.5 * 1.0))

            @staticmethod
            def mse(predicted: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
                errors = tf.subtract(predicted, labels)
                return errors, tf.square(errors)
