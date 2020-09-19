import numpy as np
import dill
from random import choices, sample
from typing import Tuple
from glob import glob
from collections import deque
from ...Utlities.exceptions import *

__all__ = ["ExperienceReplay", "PrioritizedExperienceReplay"]

class ExperienceReplay:

      def __init__(self, limit: int, batch_size: int) -> None:
          self._limit = limit
          self._batch_size = batch_size
          self._buffer = deque(maxlen=limit)
      
      def sample(self) -> np.ndarray:
          samples = sample(self._buffer, min(self._batch_size, len(self._buffer)))
          return np.array(samples), np.zeros((samples, 1), dtype=np.float32)

      def update(self, errors: np.ndarray) -> None:
          ...

      def add(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
          self._buffer.append(transition)

      def mem_size(self, buffer=self._buffer) -> int:
          mem_size = 0
          for x in buffer:
             if type(x).__name__ == 'list':
                mem_size += self.mem_size(x)
             elif type(x).__name__ == 'ndarray':
                mem_size += x.nbytes
             else:
                mem_size += sys.getsizeof(x)
          return mem_size

      def _replay(self, file: str = None, idx: int = None) -> str:
          if idx is None:
             return f'{file}.replay.*'
          return '%(file)s.replay.%(index)d' %{'file': file, 'index': idx}

      def save(self, path: str, file: str) -> None:
          mem_size = self.mem_size()
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
          replay_files = glob(os.path.join(path, self._replay()))
          if len(replay_files) == 0:
             raise MissingReplayError('Experience Replay not found')
          for file_ in replay_files:
              with open(file_, 'rb') as f_obj:
                   self._buffer.extend(dill.load(f_obj))
