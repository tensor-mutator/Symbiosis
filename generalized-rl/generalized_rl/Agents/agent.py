from abc import ABCMeta, abstractmethod
from typing import Dict

class Agent(metaclass=ABCMeta):

      @abstractmethod
      def run(self) -> None:
          ...
      
      @abstractmethod
      def _get_network_params(self, hyperparams: Dict) -> Dict:
          ...

      @abstractmethod
      def _train(self) -> float:
          ...

      @property
      @abstractmethod
      def alias(self) -> str:
          ...
      
      def _save_loss(self, loss: float) -> None:
          pass

      def _save(self) -> None:
          pass

      def _load(self) -> None:
          pass
