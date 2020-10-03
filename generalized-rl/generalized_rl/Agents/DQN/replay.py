import itertools
import dill
import sys
import os
import numpy as np
from random import choices, sample
from typing import Tuple
from glob import glob
from collections import deque
from ..Utilities.exceptions import *
from ..Utilities import Progress

__all__ = ["ExperienceReplay", "PrioritizedExperienceReplay"]

class ExperienceReplay:

      def __init__(self, limit: int, batch_size: int) -> None:
          self._limit = limit
          self._batch_size = batch_size
          self._buffer = deque(maxlen=limit)

      def sample(self) -> np.ndarray:
          n_samples = min(self._batch_size, len(self._buffer))
          samples = sample(self._buffer, n_samples)
          return np.array(samples, dtype=np.object), np.ones(shape=(n_samples, 1), dtype=np.float32)

      def update(self, errors: np.ndarray) -> None:
          ...

      def append(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
          self._buffer.append(transition)

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

      def _replay(self, file: str, idx: int = None) -> str:
          if idx is None:
             return f'{file}.replay.*'
          return '%(file)s.replay.%(index)d' %{'file': file, 'index': idx}

      def save(self, path: str, file: str) -> None:
          mem_size = self.mem_size(self._buffer)
          if mem_size == 0:
             return
          segments = mem_size//1000000
          segments = segments + 1 if mem_size%1000000 != 0 else segments
          step = len(self._buffer)//segments
          step = 1 if not step else step
          for i, x in enumerate(range(0, len(self._buffer), step)):
              with open(os.path.join(path, self._replay(file, i + 1)), 'wb') as f_obj:
                   dill.dump(deque(itertools.islice(self._buffer, x, x + step)), f_obj, protocol=dill.HIGHEST_PROTOCOL)

      def load(self, path: str, file: str) -> None:
          replay_files = glob(os.path.join(path, self._replay(file)))
          if len(replay_files) == 0:
             raise MissingReplayError('Experience Replay not found')
          for file_ in replay_files:
              with open(file_, 'rb') as f_obj:
                   self._buffer.extend(dill.load(f_obj))

class PrioritizedExperienceReplay(ExperienceReplay):

      def __init__(self, alpha: float, beta: float, offset: float,
                   limit: int, batch_size: int, progress: Progress) -> None:
          self._alpha = alpha
          self._min_beta = self._beta = beta
          self._offset = offset
          self._priorities = deque(maxlen=limit)
          self._progress = progress
          self._annealing_rate = (1 - beta)/progress.explore
          self._base = super(PrioritizedExperienceReplay, self)
          self._base.__init__(limit, batch_size)

      @property
      def beta(self) -> float:
          self._anneal_beta()
          return self._beta

      def _anneal_beta(self) -> float:
          explore_time = min(0, self._progress.clock - self._progress.observe)
          if self._beta < 1:
             self._beta = self._min_beta + self._annealing_rate*explore_time
          return self._beta

      def _sampling_probabilities(self) -> np.ndarray:
          scaled_priorities = np.array(self._priorities)**self._alpha
          sampling_probabilities = scaled_priorities/np.sum(scaled_priorities)
          return sampling_probabilities

      @property
      def _scaling_factor(self) -> float:
          scaled_priorities = np.array(self._priorities)**self._alpha
          min_probability = np.min(scaled_priorities)/np.sum(scaled_priorities)
          return (min_probability*len(self._buffer))**-self._beta

      def _importance_sampling_weights(self, sample_probabilities: np.ndarray) -> np.ndarray:
          self._anneal_beta()
          importance_sampling_weights = (sample_probabilities*len(self._buffer))**-self._beta
          return importance_sampling_weights/self._scaling_factor

      def sample(self) -> np.ndarray:
          n_samples = min(self._batch_size, len(self._buffer))
          sampling_probabilities = self._sampling_probabilities()
          self._sampling_ids = choices(range(len(self._buffer)), weights=sampling_probabilities, k=n_samples)
          samples = np.array(self._buffer, dtype=np.object)[self._sampling_ids]
          importance_sampling_weights = self._importance_sampling_weights(sampling_probabilities[self._sampling_ids])
          return samples, np.expand_dims(importance_sampling_weights, axis=1)

      def append(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
          self._base.append(transition)
          self._priorities.append(max(self._priorities, default=1))
          
      def update(self, errors: np.ndarray) -> None:
          for idx, err in zip(self._sampling_ids, errors):
              self._priorities[idx] = abs(err) + self._offset
          self._sampling_ids = None

      def _priority(self, file: str, idx: int = None) -> str:
          if idx is None:
             return f'{file}.priorities.*'
          return '%(file)s.priorities.%(index)d' %{'file': file, 'index': idx}

      def save(self, path: str, file: str) -> None:
          self._base.save(path, file)
          mem_size = self.mem_size(self._priorities)
          if mem_size == 0:
             return
          segments = mem_size//1000000
          segments = segments + 1 if mem_size%1000000 != 0 else segments
          step = len(self._priorities)//segments
          step = 1 if not step else step
          for i, x in enumerate(range(0, len(self._priorities), step)):
              with open(os.path.join(path, self._priority(file, i + 1)), 'wb') as f_obj:
                   dill.dump(deque(itertools.islice(self._priorities, x, x + step)), f_obj, protocol=dill.HIGHEST_PROTOCOL)

      def load(self, path: str, file: str) -> None:
          self._base.load(path, file)
          priority_files = glob(os.path.join(path, self._priority(file)))
          if len(priority_files) == 0:
             raise MissingReplayError('Experience Replay not found')
          for file_ in priority_files:
              with open(file_, 'rb') as f_obj:
                   self._priorities.extend(dill.load(f_obj))
