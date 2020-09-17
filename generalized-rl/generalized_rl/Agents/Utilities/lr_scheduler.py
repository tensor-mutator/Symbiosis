
__all__ = ["LRScheduler"]

class LRScheduler:

      def __init__(self, scheme: str, learning_rate: float) -> None
          self._lr = learning_rate
          self._scheme = scheme
  
