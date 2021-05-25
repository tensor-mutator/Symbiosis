import tensorflow.compat.v1 as tf
from typing import Dict

class Model:

      class Scope:

            FIT: str = "FIT"
            PREDICT: str = "PREDICT"

      @property
      def y_hat(self) -> Dict:
          return self._y_hat

      @property
      def y(self) -> Dict:
          return self._y

      @property
      def X(self) -> tf.Tensor:
          return self._X

      @property
      def grad(self) -> tf.Tensor:
          return self._grad

      @property
      def loss(self) -> float:
          return self._loss

      @property
      def scope(self) -> "<Model.Scope>":
          return Model.Scope
