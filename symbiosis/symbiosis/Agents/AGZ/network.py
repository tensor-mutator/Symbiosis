import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.regularizers as regularizers
import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple, List, Dict
from ..network_base import NetworkBaseAGZ
from ..model import Model
from ..Utilities import NetBlocks, Pipeline

class AGZChessNet(NetworkBaseAGZ):

      type: str = "AGZChess"

      class AGZChessNetModel(Model):

            def __init__(self, placeholder_X: tf.placeholder, placeholders_y: Dict = None, **params) -> None:
                self._X = placeholder_X
                p_out, v_predicted = NetBlocks.NN.ChessNet(batch_norm=True)(placeholder_X)
                logits =  NetBlocks.layers.Dense(units=action_size, kernel_regularizer=regularizers.l2(1e-4))(p_out)
                p_predicted = layers.Activation("softmax")(logits)
                self._y_hat = [p_predicted, v_predicted]
                if placeholders_y is not None:
                   self._grad = self._build_training_ops(placeholders_y, logits, **params)

            def _build_training_ops(self, placeholders_y: Dict, logits: tf.Tensor, **params) -> tf.Tensor:
                p_target, v_target = placeholders_y.get("p_target"), placeholders_y.get("v_target")
                learning_rate = params.get("learning_rate", "0.001")
                policy_error = tf.losses.softmax_cross_entropy(onehot_labels=p_target, logits=logits, weights=params.get("policy_weights", 1.25))
                value_error = tf.losses.mean_squared_error(labels=v_target, predictions=self._y_hat[1], weights=params.get("value_weights", 1.0))
                self._loss = policy_error+value_error
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                if params.get("gradient_clip_norm") is not None:
                   gradients = optimizer.compute_gradients(self._loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
                   NetBlocks.clip_grads_by_norm(gradients, params.get("gradient_clip_norm"))
                   return optimizer.apply_gradients(gradients)
                return optimizer.minimize(self._loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

      def __init__(self, state_shape: Tuple[int, int, int], action_size: int, **params) -> Pipeline:
          self._pipeline = Pipeline(meta_X=dict(shape=state_shape, dtype=tf.float32), meta_y=dict(p_target=dict(shape=action_size, dtype=tf.float32),
                                                                                                  v_target=dict(shape=(1,), dtype=tf.float32)),
                                    model=AGZChessNet.AGZChessNetModel, **params)
