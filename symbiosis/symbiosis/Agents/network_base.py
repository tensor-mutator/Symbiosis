from typing import Dict, Tuple, List
import numpy as np
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

class NetworkBaseAGZ(metaclass=NetworkMeta):

      type: str = "AGZ"

      def predict(self, X: np.ndarray) -> Tuple:
          return self._pipeline.predict(X)

      def fit(self, X: np.ndarray, ys: List[np.ndarray]) -> float:
          return self._pipeline.fit(X, ys)

      @property
      def graph(self) -> tf.Graph:
          return self._pipeline.graph

      @property
      def session(self) -> tf.Session:
          return self._pipeline.session
