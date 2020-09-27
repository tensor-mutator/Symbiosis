from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Generator, Callable
from contextlib import contextmanager
from glob import glob
import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
import numpy as np
import os
import dill
import cv2
from .DQN.replay import ExperienceReplay
from .Utilities import Progress, RewardManager, Inventory
from .Utilities.exceptions import *
from ..environment import Environment
from ..config import config

def register(suite: str) -> Callable:
    def wrapper(func) -> Callable:
        def run(inst) -> None:
            super(inst.__class__, inst).run(getattr(inst, suite))
        return run
    return wrapper

def record(func) -> Callable:
    def inner(inst, frame, *args, **kwargs) -> np.ndarray:
        if inst.config & config.SAVE_FRAMES:
           cv2.imwrite(inst._inventory.path, frame)
        return func(inst, frame, *args, **kwargs)
    return inner

class Agent(metaclass=ABCMeta):

      def __getattr__(self, func: Callable) -> Callable:
          base_idx = len(self.__class__.__mro__)-2
          agent = self.__class__.__name__
          func = self.__class__.__mro__[base_idx].__dict__.get("_{}".format(func), None)
          return lambda: func(self)  if func else None

      @contextmanager
      def _episode_context(self, env: Environment, progress: Progress,
                           reward_manager: RewardManager) -> Generator:
          env.make()
          env.reset()
          state = self.state(env.render())
          yield state
          env.close()
          reward_manager.rollout()
          if self.config & config.SAVE_WEIGHTS:
             self.save()
          progress.bump_episode()

      def _episode_suite_dqn(self) -> None:
          with self._episode_context(self.env, self.progress, self._reward_manager) as s_t:
               while not self.env.ended and self.progress.clock < self.total_steps:
                     a_t = self.action(s_t)
                     x_t1, r_t, done, _ = self.env.step(a_t)
                     self._reward_manager.update(r_t)
                     s_t1 = self.state(x_t1, s_t)
                     self.replay.add((s_t, a_t, r_t, s_t1, done,))
                     s_t = s_t1
                     if self.progress.explore_clock and self.progress.training_clock%self.training_interval == 0:
                        self.train()
                     if self.progress.explore_clock and self.progress.training_clock%self.target_frequency == 0:
                        self.update_target()
                     self.progress.bump()

      def _load_artifacts(self) -> None:
          path = self.workspace()
          if (self.config & config.LOAD_WEIGHTS) and glob(os.path.join(path,
                                                                     "{}.ckpt.*".format(self._alias))):
             self.load()
          else:
             with self.graph.as_default():
                  self.session.run(tf.global_variables_initializer())

      def run(self, suite: Callable) -> None:
          if not suite:
             raise MissingSuiteError("Matching suite not found for class: {}".format(self.__class__.__name__))
          self._reward_manager = RewardManager(self.env, self.alias, self.config, self.progress)
          if self.config & config.SAVE_FRAMES:
             self._inventory = Inventory("FRAMES", "frame", "PNG", self.env, self.alias, self.progress)
          self._load_artifacts()
          while self.progress.clock < self.total_steps:
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

      @property
      def config(self) -> config:
          return self._config

      @config.setter
      def config(self, config: config) -> None:
          self._config = config

      def save_loss(self, loss: float) -> None:
          pass

      def save(self) -> None:
          path = self.workspace()
          if not getattr(self, "_saver", None):
             with self.graph.as_default():
                  self._saver = tf.train.Saver(max_to_keep=5)
          self._saver.save(self.session, os.path.join(path, "{}.ckpt".format(self.alias)))
          with open(os.path.join(path, "{}.progress".format(self.alias)), "wb") as f_obj:
               dill.dump(self.progress, f_obj, protocol=dill.HIGHEST_PROTOCOL)
          self._reward_manager.save(path, self.alias, self._session)

      def load_progress(self) -> Progress:
          path = self.workspace()
          if (self.config & config.LOAD_WEIGHTS) and os.path.exists(os.path.join(path,
                                                                               "{}.progress".format(self.alias))):
             with open(os.path.join(path, "{}.progress".format(self.alias)), "rb") as f_obj:
                  return dill.load(f_obj)
          return Progress(self.total_steps, self.observe, self.explore)

      def load(self) -> None:
          path = self.workspace()
          with self.graph.as_default():
               self._saver = tf.train.Saver(max_to_keep=5)
          ckpt = tf.train.get_checkpoint_state(path)
          self._saver.restore(self.session, ckpt.model_checkpoint_path)
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

      @property
      def total_steps(self) -> int:
          return self._total_steps

      @property
      def observe(self) -> int:
          return getattr(self, "_observe", 0)

      @property
      def explore(self) -> float:
          return getattr(self, "_explore", np.inf)
