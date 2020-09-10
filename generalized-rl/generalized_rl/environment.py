from abc import ABCMeta, abstractmethod
from typing import Any, Sequence, Dict
import numpy as np

class Environment(metaclass=ABCMeta):

      @abstractmethod
      def make(self) -> Any:
          ...
     
      @astractmethod
      def reset(self) -> np.ndarray:
          ...

      @abstractmethod
      def step(self, action: Any) -> Sequence[np.ndarray, float, bool, Dict]:
          ...

      @abstractmethod
      def render(self) -> np.ndarray:
          ...
