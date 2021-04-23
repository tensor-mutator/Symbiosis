"""
@author: Debajyoti Raychaudhuri

A precise implementation of an AlphaGoZero agent
"""

import tensorflow.compat.v1 as tf
from .network import *
from ..agent import Agent
from ..flow_base import Flow
from ..network_base import NetworkBaseMCTS
from ..Utilities import Tree, Progress
from ...environment import Environment
from ...config import config

@Agent.track(AGZNet)
class AGZ(Agent):

      def __init__(self, env: Environment, network: NetworkBaseMCTS = AGZNet, config: bin = config.DEFAULT,
                   flow: Flow = None, **hyperparams) -> None:
          self._env = env
          self._config = config
          self._flow = flow
          self._read_params(hyperparams)
          self._alias = self._define_alias(network.type, hyperparams)
          self._progress = self.load_progress()
          self._alias = "AGZ"
          self._session = self._build_network_graph(network, hyperparams)

      def _build_network_graph(self, network: NetworkBaseMCTS, hyperparams: Dict) -> tf.Session:
          self._graph = tf.Graph()
          session = tf.Session(graph=self._graph, config=self.ConfigProto)
          with self._graph.as_default():
               optional_network_params = self._get_optional_network_params(hyperparams)
               self._local_network = network(self._env.state.shape, self._trace, self._env.action.size,
                                             **optional_network_params, scope="local")
               self._target_network = network(self._env.state.shape, self._trace, self._env.action.size,
                                              **optional_network_params, scope="target")
               self._update_ops = self._initiate_update_ops("local", "target")
          return session

      def __del__(self) -> None:
          self._session.close()
