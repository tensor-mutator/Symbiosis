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

      @abstractmethod
      def _mutate_alias(self, alias: str, hyperparams: Dict) -> str:
          ...
