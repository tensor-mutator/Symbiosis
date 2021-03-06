"""
@author: Debajyoti Raychaudhuri

A precise implementation of models to be used with AGZ algorithm
"""

import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.regularizers as regularizers
import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple, List, Dict
from ..network_base import NetworkBaseAGZ
from ..model import Model
from ..Utilities import NetBlocks, Pipeline, Metrics

class AGZChessNet(NetworkBaseAGZ):

      type: str = "AGZChess"

      class AGZChessNetModel(Model):

            def __init__(self, placeholder_X: tf.placeholder, shapes_y: Dict, placeholders_y: Dict = None, **params) -> None:
                self._X = placeholder_X
                p_out, v_predicted = NetBlocks.NN.ChessNet(batch_norm=True)(placeholder_X)
                logits =  NetBlocks.layers.Dense(units=shapes_y.get("p"),
                                                 kernel_regularizer=regularizers.l2(1e-4))(p_out)
                p_predicted = layers.Activation("softmax")(logits)
                self._y_hat = dict(p=p_predicted, v=v_predicted)
                self._y = placeholders_y
                if placeholders_y is not None:
                   self._grad = self._build_training_ops(placeholders_y, logits, **params)

            def _build_training_ops(self, placeholders_y: Dict, logits: tf.Tensor, **params) -> tf.Tensor:
                p_target, v_target = placeholders_y.get("p"), placeholders_y.get("v")
                learning_rate = params.get("learning_rate", 0.001)
                policy_error = tf.losses.softmax_cross_entropy(onehot_labels=p_target, logits=logits, weights=params.get("policy_weights", 1.25))
                value_error = tf.losses.mean_squared_error(labels=v_target, predictions=self._y_hat.get('v'), weights=params.get("value_weights", 1.0))
                self._loss = policy_error+value_error
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                if params.get("gradient_clip_norm") is not None:
                   gradients = optimizer.compute_gradients(self._loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.FIT))
                   NetBlocks.clip_grads_by_norm(gradients, params.get("gradient_clip_norm"))
                   return optimizer.apply_gradients(gradients)
                return optimizer.minimize(self._loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.FIT))

      def __init__(self, state_shape: Tuple[int, int, int], action_size: int, **params) -> None:
          self._pipeline = Pipeline(meta_X=dict(shape=state_shape, dtype=tf.float32), meta_y=dict(p=dict(shape=action_size, dtype=tf.float32,
                                                                                                         metrics=dict(MicroF1Score=Metrics.MicroF1Score,
                                                                                                                      MacroF1Score=Metrics.MacroF1Score)),
                                                                                                  v=dict(shape=1, dtype=tf.float64)),
                                    model=AGZChessNet.AGZChessNetModel, **params)
