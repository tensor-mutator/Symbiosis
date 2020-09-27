from abc import ABCMeta, abstractmethod
import numpy as np

__all__ = ["Flow"]

class Flow:

      @abstractmethod
      def flow_map(self, x_t: np.ndarray, x_t1: np.ndarray) -> np.ndarray:
          ...
