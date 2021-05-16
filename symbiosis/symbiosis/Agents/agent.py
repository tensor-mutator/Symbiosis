from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Generator, Callable, List
from contextlib import contextmanager, suppress
from glob import glob
import tensorflow.compat.v1 as tf
import numpy as np
import os
import dill
import cv2
import json
import signal
from collections import deque
from .flow_base import Flow
from .network_base import NetworkMeta
from .DQN.replay import ExperienceReplay
from .Utilities import Progress, ProgressDQN, RewardManager, Inventory
from .Utilities.exceptions import *
from ..colors import COLORS
from ..environment import Environment
from ..config import config

UNIX_SIGNALS = [signal.SIGABRT, signal.SIGBUS, signal.SIGHUP, signal.SIGILL, signal.SIGINT,
                signal.SIGQUIT, signal.SIGTERM, signal.SIGTRAP, signal.SIGTSTP]

class AgentDecorators:

      @staticmethod
      def register(suite: str) -> Callable:
          def wrapper(func: Callable) -> Callable:
              def run(inst: "<Agent inst>") -> None:
                  super(inst.__class__, inst)._run(getattr(inst, suite))
              return run
          return wrapper

      @staticmethod
      def record(func: Callable) -> Callable:
          def inner(inst: "<Agent inst>", img: np.ndarray, state: Any = None) -> List:
              path = None
              if inst.config & (config.SAVE_FRAMES+config.SAVE_FLOW):
                 if state is None:
                    path = inst._frame_inventory.init_path
                 else:
                    path = inst._frame_inventory.path
                 frame = inst.env.state.frame
                 inst._frame_buffer.append(dict(path=path, frame=frame))
                 if len(inst._frame_buffer) == inst.frame_buffer_size:
                    for frame in inst._frame_buffer:
                        cv2.imwrite(frame["path"], frame["frame"])
                    inst._frame_buffer.clear()
              return func(inst, img, state), path
          return inner

      @staticmethod
      def register_handler(unix_signals: List) -> Callable:
          def outer(handler: Callable) -> Callable:
              def inner(inst: "<Agent inst>", signal_id: int = None, frame: Any = None) -> None:
                  for sig in unix_signals:
                      signal.signal(sig, lambda x, y: handler(inst, x, y))
              return inner
          return outer

      @staticmethod
      def track(network: NetworkMeta, config: bin = config.DEFAULT, flow: Flow = None) -> Callable:
          def outer(cls: Callable) -> Callable:
              def inner(env: Environment, network: NetworkMeta = network, config: bin = config,
                        flow: Flow = flow, **hyperparams) -> None:
                  inst = cls(env, network, config, flow, **hyperparams)
                  inst._params = dict(env=env, network=network, config=config, flow=flow)
                  return inst
              return inner
          return outer

class Agent(AgentDecorators, metaclass=ABCMeta):

      def __getattr__(self, func: str) -> Callable:
          base_idx = len(self.__class__.__mro__)-3
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
          if operation == "override":
             self._override = True
             return self

      @contextmanager
      def _episode_context(self, env: Environment, progress: Progress,
                           reward_manager: RewardManager) -> Generator:
          env.reset()
          image = env.render()
          image_original = env.state.frame
          state, path = self.state(image)
          yield image, image_original, state, path
          reward_manager.rollout()
          if self.config & config.SAVE_FLOW:
             self._flow_skip_buffer.clear()
          progress.bump_episode()
          if self.progress.episode%self.checkpoint_interval==0:
             self._save()

      @contextmanager
      def _episode_context_mcts(self, env: Environment, progress: Progress) -> Generator:
          env.reset()
          _, path = self.state(env)
          yield env.state.frame, path
          if self.config & config.SAVE_FLOW:
             self._flow_skip_buffer.clear()
          progress.bump_episode()
          if self.progress.episode%self.checkpoint_interval==0:
             self._save()

      @contextmanager
      def _run_context(self, reward_manager: bool = True) -> Generator:
          self._check_hyperparams()
          self._save_hyperparams()
          self._create_handler()
          if reward_manager:
             self._reward_manager = RewardManager(self.env, self.alias, self.config, self.progress, self.writer)
          self._initiate_inventories()
          self._load()
          self._set_checkpoint_interval()
          self.env.make()
          yield
          self.env.close()
          self._save()
          self._save_all_frames()
          if self.writer:
             self.writer.close()

      def _suite_dqn(self) -> None:
          with self._run_context():
               with suppress(Exception):
                    while self.progress.clock < self.total_steps:
                          self._episode_suite_dqn()

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
                     if self.progress.train:
                        self.train()
                     if self.progress.explore_clock:
                        if (self.progress.clock-self.progress.observe)%self.target_frequency == 0:
                           self.update_target()
                     self.progress.bump()

      def _suite_agz(self) -> None:
          with self._run_context(reward_manager=False):
               with suppress(Exception):
                    while self.progress.clock < self.total_steps:
                          self._episode_suite_agz()

      def _episode_suite_agz(self) -> None:
          with self._episode_context_mcts(self.env, self.progress) as [frame_t, path_t]:
               while not self.env.ended and self.progress.clock < self.total_steps:
                     a_t = self.action(self.env)
                     _, r_t, _, _ = self.env.step(a_t)
                     _, path_t1 = self.state(env)
                     frame_t1 = self.env.state.frame
                     self._save_flow(frame_t, frame_t1, r_t, path_t, path_t1)
                     frame_t, path_t = frame_t1, path_t1
                     if self.progress.train:
                        self.train()
                     self.progress.bump()

      @AgentDecorators.register_handler(UNIX_SIGNALS)
      def _create_handler(self, signal_id: int = None, frame: Any = None):
          raise AgentInterrupt("Agent interrupted")

      def _run(self, suite: Callable) -> None:
          if not suite:
             raise MissingSuiteError("Matching suite not found for class: {}".format(self.__class__.__name__))
          suite()

      def _summary_writer(self) -> tf.summary.FileWriter:
          if self.config & config.REWARD_EVENT+config.LOSS_EVENT+config.EPSILON_EVENT+config.BETA_EVENT+config.LR_EVENT:
             return tf.summary.FileWriter(os.path.join(self.workspace, "{} EVENTS".format(self.alias)), self.graph)
          return None

      @property
      def writer(self) -> tf.summary.FileWriter:
          if getattr(self, "_writer", None) is None:
             self._writer = self._summary_writer()
          return self._writer

      def _set_checkpoint_interval(self) -> None:
          if not getattr(self, "checkpoint_interval"):
             self.checkpoint_interval = 1

      def _initiate_inventories(self) -> None:
          if self.config & (config.SAVE_FRAMES+config.SAVE_FLOW):
             if not getattr(self, "frame_buffer_size"):
                self.frame_buffer_size = 1
             self._frame_buffer = deque()
             self._frame_inventory = Inventory("FRAMES", "frame", "PNG", self.env, self.alias, self.progress)
             if self.config & config.SAVE_FLOW:
                self._flow_inventory = Inventory("FLOW", "flow", "FLO", self.env, self.alias, self.progress)
                self._buffer_len = self.flow_skip if getattr(self, "flow_skip") else 1
                self._buffer_len = max(1, self._buffer_len)
                self._flow_skip_buffer = deque(maxlen=self._buffer_len)
                self._flow_meta_buffer = deque()
                if self.flow:
                   self._flow_buffer = deque()

      def _save_flow(self, x_t: np.ndarray, x_t1: np.ndarray, r_t: float, path_t: str, path_t1: str) -> None:
          if self.config & config.SAVE_FLOW:
             self._flow_skip_buffer.append({"img": x_t, "path": path_t})
             x_t = self._flow_skip_buffer[0]["img"]
             path_t = self._flow_skip_buffer[0]["path"]
             if self.flow:
                self._flow_buffer.append(dict(path=self._flow_inventory.path, frame=self.flow.flow_map(x_t, x_t1)))
                if len(self._flow_buffer) == self.frame_buffer_size:
                   for frame in self._flow_buffer:
                       self.flow.write_flow(frame["frame"], frame["path"])
                   self._flow_buffer.clear()
             self._flow_meta_buffer.append({"flow": self._flow_inventory.path,
                                            "src_image": path_t,
                                            "dest_image": path_t1,
                                            "reward": r_t}) if self.flow else self._flow_meta_buffer.append({"src_image": path_t,
                                                                                                             "dest_image": path_t1,
                                                                                                             "reward": r_t})
             if len(self._flow_meta_buffer) == self.frame_buffer_size:
                flow_meta = list()
                if os.path.exists(os.path.join(self._flow_inventory.inventory_path, "flow.meta")):
                   with open(os.path.join(self._flow_inventory.inventory_path, "flow.meta"), "r") as f_obj:
                        flow_meta = json.load(f_obj)
                flow_meta.extend(list(self._flow_meta_buffer))
                with open(os.path.join(self._flow_inventory.inventory_path, "flow.meta"), "w") as f_obj:
                     json.dump(flow_meta, f_obj)
                self._flow_meta_buffer.clear()

      def _save_all_frames(self) -> None:
          if self.config & (config.SAVE_FRAMES+config.SAVE_FLOW):
             for frame in self._frame_buffer:
                 cv2.imwrite(frame["path"], frame["frame"])
             if self.config & config.SAVE_FLOW and self.flow:
                for frame in self._flow_buffer:
                    self.flow.write_flow(frame["frame"], frame["path"])
                flow_meta = list()
                if os.path.exists(os.path.join(self._flow_inventory.inventory_path, "flow.meta")):
                   with open(os.path.join(self._flow_inventory.inventory_path, "flow.meta"), "r") as f_obj:
                        flow_meta = json.load(f_obj)
                flow_meta.extend(list(self._flow_meta_buffer))
                with open(os.path.join(self._flow_inventory.inventory_path, "flow.meta"), "w") as f_obj:
                     json.dump(flow_meta, f_obj)

      @abstractmethod
      @AgentDecorators.register(...)
      def run(self) -> None:
          ...

      @abstractmethod
      def train(self) -> float:
          ...

      @property
      def training_interval(self) -> int:
          return self._training_interval

      @abstractmethod
      @AgentDecorators.record
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
          if self.config & config.SAVE_WEIGHTS:
             if not getattr(self, "_saver", None):
                with self.graph.as_default():
                     self._saver = tf.train.Saver(max_to_keep=5)
             self._saver.save(self.session, os.path.join(self.workspace, "{}.ckpt".format(self.alias)))
             with open(os.path.join(self.workspace, "{}.progress".format(self.alias)), "wb") as f_obj:
                  dill.dump(self.progress, f_obj, protocol=dill.HIGHEST_PROTOCOL)
          if self.config & config.REWARD_EVENT:
             self._reward_manager.save(self.workspace, self.alias, self._session)
             self.save()

      def load_progress(self, progress: Progress) -> Progress:
          if (self.config & config.LOAD_WEIGHTS) and os.path.exists(os.path.join(self.workspace,
                                                                                 "{}.progress".format(self.alias))):
             with open(os.path.join(self.workspace, "{}.progress".format(self.alias)), "rb") as f_obj:
                  return dill.load(f_obj)
          return progress(self.total_steps, self.training_interval, self.observe, self.explore)

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

      def _check_hyperparams(self) -> None:
          if not self.config & config.LOAD_WEIGHTS or getattr(self, "_override", False):
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
