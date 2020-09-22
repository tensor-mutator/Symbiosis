import tensorflow as tf
import tensorflow.keras.layers as layers
from typing import Callable, Tuple

class NetBlocks:

      graph = tf.get_default_graph()

      @staticmethod
      def op(name, scope=None):
          if scope:
             return NetBlocks.graph.get_operation_by_name('{}/{}'.format(scope, name)).outputs[0]
          return NetBlocks.graph.get_operation_by_name('{}'.format(name)).outputs[0]

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

      @staticmethod
      def Conv2DNature(batch_norm: bool = False, time_distributed: bool = False) -> Callable:
          def _op(tensor: tf.Tensor) -> tf.Tensor:
              conv1 = NetBlocks.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4),
                                         batch_norm=batch_norm, time_distributed=time_distributed)(tensor)
              conv2 = Manipulator.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                         batch_norm=batch_norm, time_distributed=time_distributed)(conv1)
              conv3 = Manipulator.Conv2D(filters=64, kernel_size=(3, 3), batch_norm=batch_norm,
                                         time_distributed=time_distributed)(conv2)
              flattened = layers.TimeDistributed(layers.Flatten())(conv3) if timedistributed else layers.Flatten()(conv3)
              return NetBlocks.Dense(units=256, batch_norm=batch_norm, time_distributed=time_distributed)(flattened)
          return _op
      
