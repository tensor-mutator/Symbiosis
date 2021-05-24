from abc import ABCMeta, abstractmethod
from typing import Any, Sequence, Dict, Tuple
import numpy as np
import json
import threading
from inspect import stack
from .Agents.Utilities import Progress

__all__ = ["Environment", "State", "Action"]

class Singleton:

      _singleton: Dict = dict()

      def __new__(cls, *args, **kwargs) -> "<Singleton inst>":
          owner = stack()[1][0].f_locals.get("self")
          key = json.dumps(dict(owner=owner.__class__.__name__, cls=cls.__name__, thread=threading.current_thread().name))
          if Singleton._singleton.get(key, None) is None:
             Singleton._singleton[key] = super(Singleton, cls).__new__(cls)
          return Singleton._singleton[key]

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

      @property
      def progress(self) -> Progress:
          return self._progress

      @progress.setter
      def progress(self, progress: Progress) -> None:
          self._progress = progress

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
