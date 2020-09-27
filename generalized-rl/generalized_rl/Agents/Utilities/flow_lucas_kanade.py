import cv2
import numpy as np
from ..flow_base import Flow

__all__ = ["LucasKanadeFlow"]

class LucasKanadeFlow(Flow):

      def flow_map(self, x_t: np.ndarray, x_t1: np.ndarray) -> np.ndarray:
          pass
