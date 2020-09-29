from abc import ABCMeta, abstractmethod
from typing import Any, Sequence, Dict, Tuple
import numpy as np

__all__ = ["Environment", "State", "Action"]

class State(metaclass=ABCMeta):

      _singleton = None

      def __new__(cls, *args, **kwargs) -> "<State inst>":
          if State._singleton is None:
             State._singleton = super(State, cls).__new__(cls, *args, **kwargs)
          return State._singleton

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

class Action(metaclass=ABCMeta):

      _singleton = None

      def __new__(cls, *args, **kwargs) -> "<State inst>":
          if Action._singleton is None:
             Action._singleton = super(Action, cls).__new__(cls, *args, **kwargs)
          return Action._singleton

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
