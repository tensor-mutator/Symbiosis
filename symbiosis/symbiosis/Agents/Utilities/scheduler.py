from typing import Callable
from abc import ABCMeta, abstractmethod
from .progress import Progress
from .exceptions import *

__all__ = ["LRScheduler", "BetaScheduler"]

class RegisterSchemes:

      @staticmethod
      def register(schemes) -> Callable:
          def outer(func) -> Callable:
              def inner(inst: Scheduler, scheme: str, _, progress: Progress) -> None:
                  if scheme not in schemes:
                     raise UnregisteredSchemeError("scheme: {} not registered with {} class".format(scheme,
                                                                                                    self.__class__.__name__))
                  inst._registered_schemes = schemes
                  inst._scheme = getattr(inst, scheme)
                  inst._progress = progress
                  return func(inst, scheme)
              return inner
          return outer

class Scheduler(RegisterSchemes, metaclass=ABCMeta):

      @abstractmethod
      @RegisterSchemes.register(...)
      def __init__(self, scheme: str, value: float, progress: Progress) -> None:
          ...

      def __getattr__(self, func: str) -> Callable:
          if func == "_registered_schemes":
             raise UnregisteredSchemeError("no scheme registration found with {} class".format(func,
                                                                                               self.__class__.__name__))
          base_idx = len(self.__class__.__mro__)-2
          if func not in self._registered_schemes:
             raise UnregisteredSchemeError("scheme: {} not registered with {} class".format(func,
                                                                                            self.__class__.__name__))
          scheme = self.__class__.__mro__[base_idx].__dict__.get("_{}".format(func), None)
          if scheme is None:
             raise UnregisteredSchemeError("invalid scheme registration : {}".format(func))
          return lambda x: scheme(self, x)
      
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
      def value(self) -> float:
          return self._scheme(self._progress.training_clock/self._progress.training_steps)

class LRScheduler(Scheduler):

      @Scheduler.register(["constant", "linear", "middle_drop", "double_linear_con", "double_middle_drop"])
      def __init__(self, scheme: str, learning_rate: float, progress: Progress) -> None:
          self._lr = learning_rate

      @property
      def lr(self) -> float:
          return self._lr*self.value

class BetaScheduler(Scheduler):

      @Scheduler.register(["constant", "linear"])
      def __init__(self, scheme: str, beta: float, progress: Progress) -> None:
          self._beta = beta

      @property
      def beta(self) -> float:
          return min(1, self._beta + (1-self._beta)*(1-self.value))
