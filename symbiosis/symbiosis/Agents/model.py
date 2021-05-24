import tensorflow.compat.v1 as tf
from typing import List

class Model:

      class Scope:

            FIT: str = "FIT"
            PREDICT: str = "PREDICT"

      @property
      def y_hat(self) -> List[tf.Tensor]:
          return self._y_hat

      @property
      def X(self) -> tf.Tensor:
          return self._X

      @property
      def y_id(self) -> List[str]:
          return self._y_id

      @property
      def grad(self) -> tf.Tensor:
          return self._grad

      @property
      def loss(self) -> float:
          return self._loss
