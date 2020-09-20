import tensorflow as tf
import tensorflow.keras.layers as layers
from typing import Callable, Tuple

class NetBlocks:

      graph = tf.get_default_graph()

      @staticmethod
      def Conv2D(filters: int, kernel_size: Tuple[int, int], strides: Tuple[int, int],
                 activation: str = "relu", batch_norm: bool = False, time_distributed: bool = False) -> Callable:
          def _op(tensor: tf.Tensor) -> tf.Tensor:
              if time_distributed:
                 tensor_out = layers.TimeDistributed(layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides))(tensor)
              else:
                 tensor_out = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(tensor)
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
      def Dense(units: int, activation: str = "relu", batch_norm: bool = False, time_distributed: bool = False) -> Callable:
          def _op(tensor: tf.Tensor) -> tf.Tensor:
              if time_distributed:
                 tensor_out = layers.TimeDistributed(layers.Dense(units=units, kernel_initializer=tf.initializers.glorot_normal()))(tensor)
              else:
                 tensor_out = layers.Dense(units=units, kernel_initializer=tf.initializers.glorot_normal())(tensor)
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
