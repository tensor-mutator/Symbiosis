import cv2
import numpy as np
from ..flow_base import Flow

__all__ = ["GunnerFarnebackFlow", "LucasKanadeFlow"]

class GunnerFarnebackFlow(Flow):

      def flow_map(self, x_t: np.ndarray, x_t1: np.ndarray) -> np.ndarray:
          flow_map = np.zeros_like(x_t)
          flow_map[..., 1] = 255
          x_t_gray = cv2.cvtColor(x_t,cv2.COLOR_BGR2GRAY)
          x_t1_gray = cv2.cvtColor(x_t1,cv2.COLOR_BGR2GRAY)
          flow = cv2.calcOpticalFlowFarneback(x_t_gray, x_t1_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
          return self.flow_to_image(flow)

class LucasKanadeFlow(Flow):

      def flow_map(self, x_t: np.ndarray, x_t1: np.ndarray) -> np.ndarray:
          raise NotImplementedError("Not implemented yet !!")
