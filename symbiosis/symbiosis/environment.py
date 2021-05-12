from abc import ABCMeta, abstractmethod
from typing import Any, Sequence, Dict, Tuple
import numpy as np
from inspect import stack

__all__ = ["Environment", "State", "Action"]

class Singleton:

      _singleton: Dict = dict()

      def __new__(cls, *args, **kwargs) -> "<Singleton inst>":
          owner = stack()[1][0].f_locals.get("self")
          if Singleton._singleton.get(owner, None) is None:
             Singleton._singleton[owner] = super(Singleton, cls).__new__(cls, *args, **kwargs)
          return Singleton._singleton[owner]

class State(Singleton, metaclass=ABCMeta):

      @property
      @abstractmethod
      def shape(self) -> Tuple:
          ...

      @property
      def frame(self) -> np.ndarray:
          return self._frame

      @frame.setter
      def frame(self, frame) -> None:
          self._frame = frame

class Action(Singleton, metaclass=ABCMeta):

      @property
      @abstractmethod
      def size(self) -> int:
          ...

class Environment(metaclass=ABCMeta):

      @property
      @abstractmethod
      def name(self) -> str:
          ...

      @abstractmethod
      def make(self) -> Any:
          ...
     
      @abstractmethod
      def reset(self) -> np.ndarray:
          ...

      @abstractmethod
      def step(self, action: Any) -> Sequence:
          ...

      @abstractmethod
      def render(self) -> np.ndarray:
          ...

      @property
      @abstractmethod
      def state(self) -> State:
          ...

      @property
      @abstractmethod
      def action(self) -> Action:
          ...

      @property
      @abstractmethod
      def ended(self) -> bool:
          ...

      @property
      @abstractmethod
      def reward(self) -> float:
          ...

      @abstractmethod
      def close(self) -> bool:
          ...
