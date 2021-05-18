"""
@author: Debajyoti Raychaudhuri

A precise implementation of an AlphaGo Zero agent
"""

import tensorflow.compat.v1 as tf
from typing import Dict
from collections import deque
import numpy as np
import dill
from .network import AGZChessNet
from ..agent import AgentMCTS, AgentForked
from ..flow_base import Flow
from ..network_base import NetworkBaseAGZ
from ..Utilities import Tree, Progress, MCTS, TauScheduler
from ...environment import Environment
from ...config import config

@Agent.track(AGZChessNet)
class AGZ(AgentMCTS):

      def __init__(self, max_min_players: Tuple[AgentForked, AgentForked], env: Environment,
                   network: NetworkBaseAGZ = AGZChessNet, config: bin = config.DEFAULT, flow: Flow = None,
                   **hyperparams) -> None:
          self._max_player, self._min_player = max_min_players
          self._env = env
          self._config = config
          self._flow = flow
          self._alias = network.type
          self._progress = self.load_progress(Progress.AGZ)
          self._session = self._build_network_graph(network)
          self._read_params(hyperparams)
          self._tau_scheduer = TauScheduler(self._tau_scheduler_scheme, self._tau_range, self._progress, self._config,
                                            self.writer)
          self._self_play_buffer = deque()

      def _read_params(self, hyperparams: Dict) -> None:
          self._tau_scheduler_scheme = hyperparams.get("tau_scheduler_scheme", "exponential")
          self._tau_range = hyperparams.get("tau_range", (0.99, 0.1))
          self._resign_value = hyperparams.get("resign_value", -0.8)
          self._min_resign_moves = hyperparams.get("min_resign_moves", 5)

      def _build_network_graph(self, network: NetworkBaseAGZ) -> tf.Session:
          graph = tf.Graph()
          session = tf.Session(graph=graph)
          with graph.as_default():
               self._network = network()
          return session

      def _predict_p_v(self, env: Environment) -> Tuple:
          p, v = self._session.run([self._network.policy, self._network.value],
                                   feed_dict={self._network.state: env.state.canonical})
          return env.predict(p, v)

      def action(self, env: Environment) -> Tuple[str, str]:
          max_action = self._max_player.action(env)
          env.step(max_action)
          min_action = self._min_player.action(env)
          return max_action, min_action

      def _policy_target(self) -> np.ndarray:
          policy = np.zeros(self._env.action.size, dtype=np.float32)
          for action, stat in self._tree[self._env.state.observation].edges.items():
              policy[self._env.action.move2index(action)] = stat.n
          policy = policy/np.sum(policy)
          return policy

      def _policy_with_temperature(self, policy: np.ndarray) -> np.ndarray:
          if self._tau_scheduler.tau < 0.1:
             policy = np.zeros(self._env.action.size, dtype=np.float32)
             policy[np.argmax(policy)] = 1
             return policy
          policy = np.power(policy, 1/self._tau_scheduler.tau)
          return policy/np.sum(policy)

      def state(self, env: Environment) -> np.ndarray:
          self._max_player.state(env)
          self._min_player.state(env)
          return env.state.frame

      @Agent.register("suite_agz")
      def run(self) -> None:
          ...

      def save(self) -> None:
          self._max_player.save(self.workspace, self._alias+self._max_player.alias)
          self._min_player.save(self.workspace, self._alias+self._min_player.alias)
          self_play_data_path = os.path.join(self.workspace, self._alias)
          with open(self_play_data_path, "wb") as io:
               dill.dump(self._self_play_buffer, io, protocol=dill.HIGHEST_PROTOCOL)

      def load(self) -> None:
          self._max_player.load(self.workspace, self._alias+self._max_player.alias)
          self._min_player.load(self.workspace, self._alias+self._min_player.alias)
          self_play_data_path = os.path.join(self.workspace, self._alias)
          if len(glob(self_play_data_path)) == 0:
             raise MissingDataError("Self play data not found")
          with open(self_play_data_path, "rb") as io:
               self._self_play_buffer = dill.load(io)

      def __del__(self) -> None:
          self._session.close()
