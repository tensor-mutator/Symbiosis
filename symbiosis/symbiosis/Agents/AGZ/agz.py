"""
@author: Debajyoti Raychaudhuri

A precise implementation of an AlphaGo Zero agent
"""

import tensorflow.compat.v1 as tf
from typing import Dict
from collections import deque
from .network import AGZChessNet
from ..agent import Agent
from ..flow_base import Flow
from ..network_base import NetworkBaseAGZ
from ..Utilities import Tree, Progress, MCTS, TauScheduler
from ...environment import Environment
from ...config import config

@Agent.track(AGZChessNet)
class AGZ(Agent):

      def __init__(self, env: Environment, network: NetworkBaseAGZ = AGZChessNet, config: bin = config.DEFAULT,
                   flow: Flow = None, **hyperparams) -> None:
          self._env = env
          self._config = config
          self._flow = flow
          self._alias = network.type
          self._progress = self.load_progress()
          self._mcts = self._initiate_tree(network, hyperparams)
          self._self_play_buffer = deque()

      def _initiate_tree(self, network: NetworkBaseAGZ, hyperparams: Dict) -> MCTS:
          virtual_loss = hyperparams.get("virtual_loss", 3)
          n_threads = hyperparams.get("n_threads", 3)
          n_simulations = hyperparams.get("n_simulations", 16)
          mcts = MCTS(env=self._env, tree=Tree(), virtual_loss=virtual_loss, n_threads=n_threads,
                      n_simulations=n_simulations, network=network, **hyperparams)
          return mcts

      def action(self, env: Environment) -> Any:
          

      @Agent.record
      def state(self, env: Environment) -> np.ndarray:
          return env.state.frame
