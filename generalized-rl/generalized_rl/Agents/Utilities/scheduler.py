from typing import Callable
from .progress import Progress
from .exceptions import *

__all__ = ["LRScheduler", "BetaScheduler"]

class Scheduler:

      def __init__(self, scheme: str, progress: Progress) -> None:
          self._scheme = getattr(self, scheme)
          self._progress = progress
          self._registered_schemes = list()

      def __getattr__(self, func: str) -> Callable:
          if func not in self._registered_schemes:
             raise UnregisteredSchemeError("scheme: {} not registered with {} class".format(func,
                                                                                            self.__class__.__name__))
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

class LRScheduler(Scheduler):

      def __init__(self, scheme: str, learning_rate: float, progress: Progress) -> None:
          super(LRScheduler, self).__init__(scheme, progress)
          self._lr = learning_rate
          self._registered_schemes = ["constant", "linear", "middle_drop", "double_linear_con", "double_middle_drop"]

      @property
      def lr(self) -> float:
          return self._lr*self._scheme(self._progress.training_clock/self._progress.training_steps)

class BetaScheduler(Scheduler):

      def __init__(self, scheme: str, beta: float, progress: Progress) -> None:
          super(BetaScheduler, self).__init__(scheme, learning_rate, progress)
          self._beta = beta
          self._registered_schemes = ["constant", "linear"]

      @property
      def beta(self) -> float:
          return self._beta + (1-self._beta)*(1-self._scheme(self._progress.training_clock/self._progress.training_steps))
