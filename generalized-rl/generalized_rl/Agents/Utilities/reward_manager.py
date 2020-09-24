from typing import List, Callable
from collections import deque
from glob import glob
import numpy as np
import tensorflow.compat.v1 as tf
import itertools
import dill
import sys
import os
from ...environment import Environment
from .exceptions import *

__all__ = ["RewardManager"]

class RewardManager:

      def __init__(self, env: Environment) -> None:
          self._env = env
          self._buffer = deque()
          self._episode_buffer = deque()
          self._n_steps = 0

      def update(self, reward: float) -> None:
          self._buffer.append(reward)
          self._n_steps += 1

      def _total_rewards_and_steps(self) -> List:
          rewards = 0
          steps = 0
          for episode in self._episode_buffer:
              rewards += episode["total"]
              steps += episode["steps"]
          return rewards, steps

      def rollout(self) -> None:
          rewards = np.array(self._buffer)
          mean_reward = np.mean(rewards)
          median_reward = np.median(rewards)
          max_reward = np.max(rewards)
          min_reward = np.min(rewards)
          total_rewards = np.sum(rewards)
          total_rewards_prior, n_steps_prior = self._total_rewards_and_steps()
          cumulative_mean_reward = (total_rewards_prior + total_rewards)/(n_steps_prior + self._n_steps)
          payload = dict(mean=mean_reward, median=median_reward, max=max_reward, min=min_reward,
                         total=total_rewards, cumulative_mean=cumulative_mean_reward, steps=self._n_steps)
          self._episode_buffer.append(payload)
          self._buffer = deque()
          self._n_steps = 0

      def mem_size(self, buffer: deque) -> int:
          mem_size = 0
          for x in buffer:
             if type(x).__name__ == 'list':
                mem_size += self.mem_size(x)
             elif type(x).__name__ == 'ndarray':
                mem_size += x.nbytes
             else:
                mem_size += sys.getsizeof(x)
          return mem_size

      def _episodic_reward(self, file: str = None, idx: int = None) -> str:
          if idx is None:
             return f'{file}.episode.reward.*'
          return '%(file)s.episode.reward.%(index)d' %{'file': file, 'index': idx}

      def _reward(self, file: str = None, idx: int = None) -> str:
          if idx is None:
             return f'{file}.reward.*'
          return '%(file)s.reward.%(index)d' %{'file': file, 'index': idx}

      def _create_tensorflow_event_file(self, path: str, agent: str, graph: tf.Graph) -> None:
          summary_writer = tf.summary.FileWriter(path, graph)
          for idx, episode in enumerate(self._episode_buffer):
              total, max, min, median, mean, cumulative_mean, steps = [tf.Summary() for _ in range(7)]
              total.value.add(tag='Performance Benchmark on {}/Episodes - Total Rewards'.format(self._env.name),
                              simple_value=episode["total"])
              max.value.add(tag='Performance Benchmark on {}/Episodes - Max Rewards'.format(self._env.name),
                              simple_value=episode["max"])
              min.value.add(tag='Performance Benchmark on {}/Episodes - Min Rewards'.format(self._env.name),
                              simple_value=episode["min"])
              median.value.add(tag='Performance Benchmark on {}/Episodes - Median Rewards'.format(self._env.name),
                              simple_value=episode["median"])
              mean.value.add(tag='Performance Benchmark on {}/Episodes - Mean Rewards'.format(self._env.name),
                              simple_value=episode["mean"])
              cumulative_mean.value.add(tag='Performance Benchmark on {}/Episodes - Cumulative Mean Rewards'.format(self._env.name),
                              simple_value=episode["cumulative_mean"])
              steps.value.add(tag='Performance Benchmark on {}/Episodes - Steps'.format(self._env.name),
                              simple_value=episode["steps"])
              [summary_writer.add_summary(x, idx+1) for x in [total, max, min, median, mean, cumulative_mean, steps]]

      def save(self, path: str, file: str, session: tf.Session) -> None:
          def _save(obj: deque, func: Callable):
              mem_size = self.mem_size(obj)
              if mem_size == 0:
                 return
              segments = mem_size//1000000
              segments = segments+1 if mem_size%1000000 != 0 else segments
              step = len(obj)//segments
              step = 1 if not step else step
              for i, x in enumerate(range(0, len(obj), step)):
                  with open(os.path.join(path, func(file, i+1)), 'wb') as f_obj:
                       dill.dump(deque(itertools.islice(obj, x, x+step)), f_obj, protocol=dill.HIGHEST_PROTOCOL)
          _save(self._episode_buffer, self._episodic_reward)
          _save(self._buffer, self._reward)
          self._create_tensorflow_event_file(path, file, session.graph)

      def load(self, path: str, file: str) -> None:
          def _load(obj: deque, func: Callable, raise_: bool = True):
              files = glob(os.path.join(path, func()))
              if len(files) == 0:
                 if raise_:
                    raise MissingRewardArtifactError("Reward Artifact not found")
              for file_ in files:
                  with open(file_, 'rb') as f_obj:
                       obj.extend(dill.load(f_obj))
          _load(self._episode_buffer, self._episodic_reward)
          _load(self._buffer, self._reward, raise_=False)
          self._n_steps = len(self._buffer)
