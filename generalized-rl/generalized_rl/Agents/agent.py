from abc import ABCMeta, abstractmethod
from typing import Dict, Any
import tensorflow as tf
import os
from .utilities import Progress

class Agent(metaclass=ABCMeta):

      @abstractmethod
      def run(self) -> None:
          ...

      @abstractmethod
      def train(self) -> float:
          ...

      @property
      @abstractmethod
      def action(self) -> Any:
          ...
      
      @property
      def session(self) -> tf.Session:
          return self._session

      @property
      def graph(self) -> tf.Graph:
          retrun self._graph

      @property
      def alias(self) -> str:
          return self._alias

      @property
      def progress(self) -> Progress:
          return self._progress

      @property
      def env(self) -> str:
          return self._env.name
      
      def save_loss(self, loss: float) -> None:
          pass

      def save(self) -> None:
          with self._graph.as_default():
               saver = tf.train.Saver(max_to_keep=5)
          saver.save(self.session, os.path.join(self.workspace(), "{}.ckpt".format(self.alias)))

      def load(self) -> None:
          with self._graph.as_default():
               saver = tf.train.Saver(max_to_keep=5)
          ckpt = tf.train.get_checkpoint_state(self.workspace())
          saver.restore(self.session, ckpt.model_checkpoint_path)

      def workspace(self) -> str:
          path = os.path.join(self.env, self.alias)
          if not os.path.exists(path):
             os.makedirs(path)
          return path
