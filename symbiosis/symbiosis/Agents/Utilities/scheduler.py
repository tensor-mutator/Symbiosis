from typing import Callable, Tuple, List
import tensorflow.compat.v1 as tf
from .progress import Progress
from .exceptions import *
from .event_writer import EventWriter
from ...config import config

__all__ = ["LRScheduler", "BetaScheduler", "EpsilonGreedyScheduler"]

class RegisterSchemes:

      @staticmethod
      def register(schemes: List) -> Callable:
          def outer(cls: Callable) -> Callable:
              def inner(scheme: str, value: float, progress: Progress, config_: config,
                        writer: tf.summary.FileWriter) -> None:
                  if scheme not in schemes:
                     raise UnregisteredSchemeError("scheme: {} not registered with {} class".format(scheme,
                                                                                                    cls.__name__))
                  inst = cls(scheme, value, progress, config, writer)
                  inst._registered_schemes = schemes
                  inst._scheme = getattr(inst, scheme)
                  return inst
              return inner
          return outer

class Scheduler(RegisterSchemes):

      def __getattr__(self, func: str) -> Callable:
          if func == "_registered_schemes":
             raise UnregisteredSchemeError("no scheme registration found with {} class".format(func,
                                                                                               self.__class__.__name__))
          base_idx = len(self.__class__.__mro__)-3
          if func not in self._registered_schemes:
             raise UnregisteredSchemeError("scheme: {} not registered with {} class".format(func,
                                                                                            self.__class__.__name__))
          scheme = self.__class__.__mro__[base_idx].__dict__.get("_{}".format(func), None)
          if scheme is None:
             raise UnregisteredSchemeError("invalid scheme registration : {}".format(func))
          return lambda *args, **kwargs: scheme(self, *args, **kwargs)
      
      def _constant(self, p: float) -> float:
          return 1

      def _linear(self, p: float) -> float:
          return 1-p

      def _exponential(self, p: float, decay_factor: float) -> float:
          return (1-decay_factor)**p

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

      def value(self, p: float, *args, **kwargs) -> float:
          return self._scheme(p, *args, **kwargs)

@Scheduler.register(["constant", "linear", "middle_drop", "double_linear_con", "double_middle_drop"])
class LRScheduler(EventWriter, Scheduler):

      def __init__(self, scheme: str, learning_rate: float, progress: Progress, config_: config,
                   writer: tf.summary.FileWriter) -> None:
          self._lr = learning_rate
          self._progress = progress
          self._set_writer('Hyperparams Schedule/Epoch - Learning Rate', config_ & config.LR_EVENT,
                           writer, progress, "training_clock")

      @EventWriter.registerwriter
      @property
      def lr(self) -> float:
          return self._lr*self.value(self._progress.training_clock/self._progress.training_steps)

@Scheduler.register(["constant", "linear"])
class BetaScheduler(EventWriter, Scheduler):

      def __init__(self, scheme: str, beta: float, progress: Progress, config_: config,
                   writer: tf.summary.FileWriter) -> None:
          self._beta = beta
          self._progress = progress
          self._set_writer('Hyperparams Schedule/Epoch - Beta', config_ & config.BETA_EVENT,
                           writer, progress, "training_clock")

      @EventWriter.registerwriter
      @property
      def beta(self) -> float:
          return min(1, self._beta + (1-self._beta)*(1-self.value(self._progress.training_clock/self._progress.training_steps)))

@Scheduler.register(["constant", "linear", "exponential"])
class EpsilonGreedyScheduler(EventWriter, Scheduler):

      def __init__(self, scheme: str, epsilon_range: Tuple[float, float], progress: Progress, config_: config,
                   writer: tf.summary.FileWriter) -> None:
          self._epsilon = epsilon_range[0]
          self._progress = progress
          self._scheme = scheme
          if scheme == "exponential":
             self._decay_factor = 1-epsilon_range[1]
          self._set_writer('Hyperparams Schedule/Steps - Epsilon', config_ & config.EPSILON_EVENT,
                           writer, progress, "clock")

      @EventWriter.registerwriter
      @property
      def epsilon(self) -> float:
          if self._scheme == "exponential":
             multiplier = self.value(self._progress.explore_clock/self._progress.explore, decay_factor=self._decay_factor)
          else:
             multiplier = self.value(self._progress.explore_clock/self._progress.explore)
          return self._epsilon*multiplier
