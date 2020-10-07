from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Generator, Callable
from contextlib import contextmanager, suppress
from glob import glob
import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
import numpy as np
import os
import dill
import cv2
import json
import signal
from collections import deque
from .decorators import record, register_handler, track
from .flow_base import Flow
from .network_base import NetworkMeta
from .DQN.replay import ExperienceReplay
from .Utilities import Progress, RewardManager, Inventory
from .Utilities.exceptions import *
from ..colors import COLORS
from ..environment import Environment
from ..config import config

UNIX_SIGNALS = [signal.SIGABRT, signal.SIGBUS, signal.SIGHUP, signal.SIGILL, signal.SIGINT,
                signal.SIGQUIT, signal.SIGTERM, signal.SIGTRAP, signal.SIGTSTP]

class Agent(metaclass=ABCMeta):

      @abstractmethod
      @track(...)
      def __init__(self, *args, **kwargs) -> None:
          ...

      def __getattr__(self, func: str) -> Callable:
          base_idx = len(self.__class__.__mro__)-2
          if func == "_save":
             return self.save_
          if func == "_load":
             return self.load_
          func = self.__class__.__mro__[base_idx].__dict__.get("_{}".format(func), None)
          if func:
             return lambda: func(self)
          return None

      def __call__(self, operation: str) -> "<Agent inst>":
          if operation == "restore":
             if self._hyperparams_file:
                self._params.update(self._old_params)
                inst = self.__class__(**self._params)
                return inst
             return self
          else:
             self._continue = True
             return self

      @contextmanager
      def _episode_context(self, env: Environment, progress: Progress,
                           reward_manager: RewardManager) -> Generator:
          env.make()
          env.reset()
          image = env.render()
          image_original = env.state.frame
          state, path = self.state(image)
          yield image, image_original, state, path
          env.close()
          reward_manager.rollout()
          if self.config & config.SAVE_FLOW:
             self._flow_buffer.clear()
          progress.bump_episode()
          if self.progress.episode%self.checkpoint_interval==0:
             self._save()

      @contextmanager
      def _run_context(self) -> Generator:
          self._check_hyperparams(getattr(self, "_continue", False))
          self._save_hyperparams()
          self._create_handler()
          self._writer = self._summary_writer()
          self._reward_manager = RewardManager(self.env, self.alias, self.config, self.progress, self._writer)
          self._initiate_inventories()
          self._load()
          if not getattr(self, "checkpoint_interval"):
             self.checkpoint_interval = 1
          yield
          self._save()
          self._save_all_frames()
          if self._writer:
             self._writer.close()

      def _episode_suite_dqn(self) -> None:
          with self._episode_context(self.env, self.progress, self._reward_manager) as [x_t, frame_t, s_t, path_t]:
               while not self.env.ended and self.progress.clock < self.total_steps:
                     a_t = self.action(s_t)
                     x_t1, r_t, done, _ = self.env.step(a_t)
                     self._reward_manager.update(r_t)
                     s_t1, path_t1 = self.state(x_t1, s_t)
                     frame_t1 = self.env.state.frame
                     self._save_flow(frame_t, frame_t1, r_t, path_t, path_t1)
                     self.replay.append((s_t, a_t, r_t, s_t1, done,))
                     x_t, frame_t, s_t, path_t = x_t1, frame_t1, s_t1, path_t1
                     if self.progress.explore_clock:
                        if self.progress.training_clock%self.training_interval == 0:
                           self.train()
                        if self.progress.training_clock%self.target_frequency == 0:
                           self.update_target()
                     self.progress.bump()

      @register_handler(UNIX_SIGNALS)
      def _create_handler(self, signal_id: int = None, frame: Any = None):
          raise AgentInterrupt("Agent interrupted")

      def run(self, suite: Callable) -> None:
          if not suite:
             raise MissingSuiteError("Matching suite not found for class: {}".format(self.__class__.__name__))
          with self._run_context():
               with suppress(Exception):
                    while self.progress.clock < self.total_steps:
                          suite()

      def _summary_writer(self) -> tf.summary.FileWriter:
          if self.config & config.REWARD_EVENT+config.LOSS_EVENT+config.EPSILON_EVENT+config.BETA_EVENT+config.LR_EVENT:
             return tf.summary.FileWriter(os.path.join(self.workspace, "{} EVENTS".format(self.alias)), self.graph)
          return None

      def _initiate_inventories(self) -> None:
          if self.config & (config.SAVE_FRAMES+config.SAVE_FLOW):
             if not getattr(self, "frame_buffer_size"):
                self.frame_buffer_size = 1
             self._frame_buffer = deque()
             self._frame_inventory = Inventory("FRAMES", "frame", "PNG", self.env, self.alias, self.progress)
             if self.config & config.SAVE_FLOW:
                self._flow_inventory = Inventory("FLOW", "flow", "PNG", self.env, self.alias, self.progress)
                self._buffer_len = self.flow_skip if getattr(self, "flow_skip") else 1
                self._buffer_len = max(1, self._buffer_len)
                self._flow_buffer = deque(maxlen=self._buffer_len)
                if self.flow:
                   self._frame_buffer_flow = deque()

      def _save_flow(self, x_t: np.ndarray, x_t1: np.ndarray, r_t: float, path_t: str, path_t1: str) -> None:
          if self.config & config.SAVE_FLOW:
             self._flow_buffer.append({"img": x_t, "path": path_t})
             x_t = self._flow_buffer[0]["img"]
             path_t = self._flow_buffer[0]["path"]
             if self.flow:
                self._frame_buffer_flow.append(dict(path=self._flow_inventory.path, frame=self.flow.flow_map(x_t, x_t1)))
                if len(self._frame_buffer_flow) == self.frame_buffer_size:
                   for frame in self._frame_buffer_flow:
                       cv2.imwrite(frame["path"], frame["frame"])
                   self._frame_buffer_flow.clear()
             flow_meta = list()
             if os.path.exists(os.path.join(self._flow_inventory.inventory_path, "flow.meta")):
                with open(os.path.join(self._flow_inventory.inventory_path, "flow.meta"), "r") as f_obj:
                     flow_meta = json.load(f_obj)
             flow_meta.append({"flow": self._flow_inventory.path,
                               "src_image": path_t,
                               "dest_image": path_t1,
                               "reward": r_t}) if self.flow else flow_meta.append({"src_image": path_t,
                                                                                   "dest_image": path_t1,
                                                                                   "reward": r_t})
             with open(os.path.join(self._flow_inventory.inventory_path, "flow.meta"), "w") as f_obj:
                  json.dump(flow_meta, f_obj)

      def _save_all_frames(self) -> None:
          if self.config & (config.SAVE_FRAMES+config.SAVE_FLOW):
             for frame in self._frame_buffer:
                 cv2.imwrite(frame["path"], frame["frame"])
             if self.config & config.SAVE_FLOW and self.flow:
                for frame in self._frame_buffer_flow:
                    cv2.imwrite(frame["path"], frame["frame"])

      @abstractmethod
      def train(self) -> float:
          ...

      @property
      def training_interval(self) -> int:
          return self._training_interval

      @abstractmethod
      @record
      def state(self, x_t1: np.ndarray, s_t: np.ndarray = None) -> np.ndarray:
          ...

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

      @abstractmethod
      def save(self) -> None:
          ...

      def save_(self) -> None:
          if not self.config & config.SAVE_WEIGHTS:
             return
          if not getattr(self, "_saver", None):
             with self.graph.as_default():
                  self._saver = tf.train.Saver(max_to_keep=5)
          self._saver.save(self.session, os.path.join(self.workspace, "{}.ckpt".format(self.alias)))
          with open(os.path.join(self.workspace, "{}.progress".format(self.alias)), "wb") as f_obj:
               dill.dump(self.progress, f_obj, protocol=dill.HIGHEST_PROTOCOL)
          self._reward_manager.save(self.workspace, self.alias, self._session)
          self.save()

      def load_progress(self) -> Progress:
          if (self.config & config.LOAD_WEIGHTS) and os.path.exists(os.path.join(self.workspace,
                                                                                 "{}.progress".format(self.alias))):
             with open(os.path.join(self.workspace, "{}.progress".format(self.alias)), "rb") as f_obj:
                  return dill.load(f_obj)
          return Progress(self.total_steps, self.observe, self.explore)

      def _save_hyperparams(self) -> None:
          if not self.config & config.SAVE_WEIGHTS:
             return
          with open(os.path.join(self.workspace, "{}.hyperparams".format(self.alias)), "w") as f_obj:
               json.dump(self.hyperparams, f_obj, indent=2)

      @property
      def _old_params(self) -> Dict:
          with open(os.path.join(self.workspace, "{}.hyperparams".format(self.alias)), "r") as f_obj:
               return json.load(f_obj)

      @property
      def _hyperparams_file(self) -> bool:
          exists = False
          if glob(os.path.join(self.workspace, "{}.*".format(self.alias))):
             if not os.path.exists(os.path.join(self.workspace, "{}.hyperparams".format(self.alias))):
                raise MissingHyperparamsError("{} file not found".format(os.path.join(self.workspace,
                                                                                      "{}.hyperparams".format(self.alias))))
             exists = True
          return exists

      def _check_hyperparams(self, continue_: bool) -> None:
          if not self.config & config.LOAD_WEIGHTS or continue_:
             return
          if self._hyperparams_file:
             hyperparams = self._old_params
             if hyperparams != self.hyperparams:
                msg = f"\n{COLORS.RED}** hyperparameters are different from the ones that are used during the last run **\n\n"
                for param, value in self.hyperparams.items():
                    old_value = hyperparams.get(param)
                    if value != old_value:
                       msg += f"{COLORS.MAGENTA}{param} = {COLORS.RED}{value} {COLORS.MAGENTA}!= {old_value}{COLORS.DEFAULT}\n"
                raise HyperparamsMismatchError(msg)

      @abstractmethod
      def load(self) -> None:
          ...

      def load_(self) -> None:
          if not (self.config & config.LOAD_WEIGHTS) or not glob(os.path.join(self.workspace,
                                                                              "{}.ckpt.*".format(self.alias))):
             with self.graph.as_default():
                  self.session.run(tf.global_variables_initializer())
             return
          with self.graph.as_default():
               self._saver = tf.train.Saver(max_to_keep=5)
          ckpt = tf.train.get_checkpoint_state(self.workspace)
          self._saver.restore(self.session, ckpt.model_checkpoint_path)
          self._reward_manager.load(self.workspace, self.alias)
          self.load()

      @property
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

      @property
      def flow(self) -> Flow:
          return self._flow

      @property
      def hyperparams(self) -> Dict:
          return self._hyperparams

      @property
      def ConfigProto(self) -> tf.ConfigProto:
          config = tf.ConfigProto()
          config.gpu_options.allow_growth = True
          return config
