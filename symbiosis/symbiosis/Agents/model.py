import tensorflow.compat.v1 as tf

class Model:

      @property
      def y_hat(self) -> List[tf.Tensor]:
          return self._y_hat

      @property
      def X(self) -> tf.Tensor:
          return self._X

      @property
      def grad(self) -> tf.Tensor:
          return self._grad

      @property
      def loss(self) -> float:
          return self._loss
