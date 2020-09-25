from typing import Callable
from .progress import Progress

__all__ = ["LRScheduler"]

class LRScheduler:

      def __init__(self, scheme: str, learning_rate: float, training_steps: int,
                   progress: Progress) -> None:
          self._lr = learning_rate
          self._scheme = getattr(self, scheme)
          self._n_steps = training_steps
          self._progress = progress

      def __getattr__(self, func: str) -> Callable:
          return lambda x: self.__class__.__dict__.get("_{}".format(func))(self, x)
      
      def _constant(self, p: float) -> float:
          return 1

      def _linear(self, p: float) -> float:
          return 1-p

      def _middle_drop(self, p: float) -> float:
          eps = 0.75
          if 1-p < eps:
            return eps*0.1
          return 1-p

      def _double_linear_con(self, p: float) -> float:
          p *= 2
          eps = 0.125
          if 1-p < eps:
             return eps
          return 1-p

      def _double_middle_drop(self, p: float) -> float:
          eps1 = 0.75
          eps2 = 0.25
          if 1-p < eps1:
             if 1-p < eps2:
                return eps2*0.5
             return eps1*0.1
          return 1-p

      @property
      def lr(self) -> float:
          return self._lr*self._scheme(self._progress.training_clock/self._n_steps)
