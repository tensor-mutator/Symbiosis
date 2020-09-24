from typing import Dict, Tuple
import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf

class NetworkMeta(type):

      def __new__(cls, name: str, base: Tuple, body: Dict) -> "<NetworkMeta inst>":
          if "type" not in list(body.keys()):
             raise TypeError("Missing type attribute")
          return super(NetworkMeta, cls).__new__(cls, name, base, body)

class NetworkBaseDQN(metaclass=NetworkMeta):

      type: str = "DQN"

      @property
      def state(self) -> tf.Tensor:
          return self._state

      @property
      def action(self) -> tf.Tensor:
          return self._action

      @property
      def q_predicted(self) -> tf.Tensor:
          return self._q_predicted

      @property
      def q_target(self) -> tf.Tensor:
          return self._q_target

      @property
      def importance_sampling_weights(self) -> tf.Tensor:
          return self._importance_sampling_weights

      @property
      def learning_rate(self) -> float:
          return self._learning_rate

      @property
      def grad(self) -> tf.Tensor:
          return self._grad

      @property
      def error(self) -> tf.Tensor:
          return self._error

      @property
      def loss(self) -> tf.Tensor:
          return self._loss

