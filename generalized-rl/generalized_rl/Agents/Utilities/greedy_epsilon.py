from typing import Tuple, Callable
from .progress import Progress

__all__ = ["GreedyEpsilon"]

class GreedyEpsilon:

      def __init__(self, progress: Progress, epsilon_range: Tuple = (1, 0.0001), scheme: str = "linear") -> None:
          self._max_epsilon, self._min_epsilon = epsilon_range
          self._scheme = getattr(self, scheme)
          self._progress = progress
          self._epsilon = self._max_epsilon
          self._linear_rate = (self._max_epsilon - self._min_epsilon)/progress.explore

      def __getattr__(self, func) -> Callable:
          return lambda: self.__class__.__dict__.get("_{}".format(func))(self)

      @property
      def epsilon(self) -> float:
          return self._epsilon

      def _linear(self) -> None:
          explore_time = self._progress.explore_clock
          self._epsilon = self._max_epsilon - self._linear_rate*explore_time
              
      def _exponential(self) -> None:
          const = 10/self._progress.explore
          explore_time = self._progress.explore_clock
          self._epsilon = self._min_epsilon + ((self._max_epsilon - self._min_epsilon)*np.exp(-const*explore_time))

      def decay(self) -> None:
          self._scheme()
