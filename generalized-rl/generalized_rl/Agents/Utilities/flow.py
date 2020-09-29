import cv2
import numpy as np
from ..flow_base import Flow

__all__ = ["GunnerFarnebackFlow", "LucasKanadeFlow"]

class GunnerFarnebackFlow(Flow):

      def flow_map(self, x_t: np.ndarray, x_t1: np.ndarray) -> np.ndarray:
          flow_map = np.zeros_like(frame1)
          flow_map[...,1] = 255
          x_t_gray = cv2.cvtColor(x_t,cv2.COLOR_BGR2GRAY)
          x_t1_gray = cv2.cvtColor(x_t1,cv2.COLOR_BGR2GRAY)
          flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
          mag, ang = cv2.cartToPolar(flow_map[...,0], flow_map[...,1])
          flow_map[...,0] = ang*180/np.pi/2
          flow_map[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
          return cv2.cvtColor(flow_map,cv2.COLOR_HSV2BGR)

class LucasKanadeFlow(Flow):

      def flow_map(self, x_t: np.ndarray, x_t1: np.ndarray) -> np.ndarray:
          pass
