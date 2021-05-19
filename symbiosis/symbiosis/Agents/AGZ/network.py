import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.regularizers as regularizers
import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple, List
from ..network_base import NetworkBaseAGZ
from ..Utilities import NetBlocks, Pipeline

class AGZChessNet(NetworkBaseAGZ):

      type: str = "AGZChess"

      def __init__(self, state_shape: Tuple[int, int, int], action_size: int, **params) -> Pipeline:
          placeholders = dict(state=NetBlocks.placeholder(state_shape), p_target=NetBlocks.placeholder(shape=[action_size]),
                              v_target=NetBlocks.placeholder(shape=[], dtype=tf.float32))
          return Pipeline(placeholders, self._model)

      def _model(placeholders: Dict) -> None:
          self._state = placeholders.get("state")
          p_out, self._v_predicted = NetBlocks.NN.ChessNet(batch_norm=True)(self._state)
          logits =  NetBlocks.layers.Dense(units=action_size, kernel_regularizer=regularizers.l2(1e-4))(p_out)
          self._p_predicted = layers.Activation("softmax")(logits)
          self._grad = self._build_training_ops(placeholders, logits, **params)

      def _build_training_ops(self, placeholders: Dict, logits: tf.Tensor, **params) -> tf.Tensor:
          self._p_target, self._v_target = placeholders.get("p_target"), placeholders.get("v_target")
          self._learning_rate = NetBlocks.placeholder(shape="scalar")
          policy_error = tf.losses.softmax_cross_entropy(onehot_labels=self._p_target, logits=logits, weights=params.get("policy_weights", 1.25))
          value_error = tf.losses.mean_squared_error(labels=self._v_target, predictions=self._v_predicted, weights=params.get("value_weights", 1.0))
          self._loss = policy_error+value_error
          optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
          if params.get("gradient_clip_norm") is not None:
             gradients = optimizer.compute_gradients(self._loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
             NetBlocks.clip_grads_by_norm(gradients, params.get("gradient_clip_norm"))
             return optimizer.apply_gradients(gradients)
          return optimizer.minimize(self._loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
