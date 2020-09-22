from abc import ABCMeta, abstractmethod
import tensorflow as tf

class NetworkBaseDQN:

      name: str = "DQN"

      @property
      def state(self) -> tf.Tensor:
          return self._state

      @property
      def action(self) -> tf.Tensor:
          return self._action

      @property
      def q_predicted(self) -> tf.Tensor:
          return self._predicted

      @property
      def q_target(self) -> tf.Tensor:
          return self._q_target

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
