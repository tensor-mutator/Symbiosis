from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Generator, Callable
from contextlib import contextmanager
from glob import glob
import tensorflow.compat.v1 as tf
import numpy as np
import os
import dill
from .DQN.replay import ExperienceReplay
from .Utilities import Progress, RewardManager
from .Utilities.exceptions import *
from ..environment import Environment

def register(suite: str) -> Callable:
    def wrapper(func) -> Callable:
        def run(self) -> None:
            super(self.__class__, self).run(getattr(self, suite))
        return run
    return wrapper

class Agent(metaclass=ABCMeta):

      def __getattr__(self, func: Callable) -> Callable:
          agent = self.__class__.__name__
          if getattr(self, "_{}".format(func), None):
             return lambda: self.__class__.__mro__[1].__dict__["_{}".format(func)](self)
          raise MissingSuiteError("Matching suite not found for class: {}".format(agent))

      @contextmanager
      def _episode_context(self, env: Environment, progress: Progress,
                           reward_manager: RewardManager) -> Generator:
          env.make()
          env.reset()
          state = self.state(env.render())
          yield env, progress, reward_manager, state
          progress.bump_episode()
          env.close()
          reward_manager.rollout()
          self.save()

      def _episode_suite_dqn(self) -> None:
          with self._episode_context(self._env, self._progress, self._reward_manager) as env, progress, reward_manager, s_t:
               while not env.ended:
                     a_t = self.action(s_t)
                     x_t1, r_t, done, _ = env.step(a_t)
                     s_t1 = self.state(x_t1, s_t)
                     self.replay.add((s_t, a_t, r_t, s_t1, done,))
                     s_t = s_t1
                     if progress.explore_clock and progress.explore_clock%self.training_interval == 0:
                        self.train()
                     if progress.explore_clock and progress.explore_clock%self.target_frequency == 0:
                        self.update_target()
                     progress.bump()

      def _load_artifacts(self) -> None:
          path = self.workspace()
          if glob(os.path.join(path, "{}.ckpt.*".format(self._alias))):
             self.load()
          else:
             with self.session.graph.as_default():
                  self.session.run(tf.global_variables_initializer())

      def run(self, suite: Callable) -> None:
          self._reward_manager = RewardManager(self._env)
          self._load_artifacts()
          while True:
                suite()

      @abstractmethod
      def train(self) -> float:
          ...

      @property
      def training_interval(self) -> int:
          return self._training_interval

      @abstractmethod
      def action(self, state: np.ndarray) -> Any:
          ...
      
      @property
      def session(self) -> tf.Session:
          return self._session

      @property
      def graph(self) -> tf.Graph:
          return self._graph

      @property
      def alias(self) -> str:
          return self._alias

      @property
      def replay(self) -> ExperienceReplay:
          return self._replay

      @property
      def progress(self) -> Progress:
          return self._progress

      @property
      def env(self) -> str:
          return self._env
      
      def save_loss(self, loss: float) -> None:
          pass

      def save(self) -> None:
          path = self.workspace()
          with self._graph.as_default():
               saver = tf.train.Saver(max_to_keep=5)
          saver.save(self.session, os.path.join(path, "{}.ckpt".format(self.alias)))
          with open(os.path.join(path, "{}.progress".format(self.alias)), "wb") as f_obj:
               dill.dump(self._progress, f_obj, protocol=dill.HIGHEST_PROTOCOL)
          self._reward_manager.save(path, self.alias, self._session)

      def load_progress(self) -> Progress:
          path = self.workspace()
          if os.path.exists(os.path.join(path, "{}.progress".format(self.alias))):
             with open(os.path.join(path, "{}.progress".format(self.alias)), "rb") as f_obj:
                  return dill.load(f_obj)
          return Progress(self._observe, self._explore)

      def load(self) -> None:
          with self._graph.as_default():
               saver = tf.train.Saver(max_to_keep=5)
          ckpt = tf.train.get_checkpoint_state(self.workspace())
          saver.restore(self.session, ckpt.model_checkpoint_path)
          self._reward_manager.load(path, self.alias)

      def workspace(self) -> str:
          path = os.path.join(self.env.name, self.alias)
          if not os.path.exists(path):
             os.makedirs(path)
          return path

      @property
      def update_ops(self) -> tf.group:
          return self._update_ops

      def update_target(self) -> None:
          self.session.run(self.update_ops)

      @property
      def target_frequency(self) -> int:
          return self._target_frequency
