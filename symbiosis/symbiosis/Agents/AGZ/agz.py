"""
@author: Debajyoti Raychaudhuri

A precise implementation of an AlphaGo Zero agent
"""

import tensorflow.compat.v1 as tf
from typing import Dict, Any
from collections import deque
import numpy as np
import dill
from random import sample
from .network import AGZChessNet
from ..agent import AgentMCTS, AgentForked
from ..flow_base import Flow
from ..network_base import NetworkBaseAGZ
from ..Utilities import Progress
from ...environment import Environment
from ...config import config

@Agent.track(AGZChessNet)
class AGZ(AgentMCTS):

      def __init__(self, max_min_players: Tuple[AgentForked, AgentForked], env: Environment,
                   network: NetworkBaseAGZ = AGZChessNet, config: bin = config.DEFAULT, flow: Flow = None) -> None:
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
          self._initiate_players(self._predict_p_v, self._self_play_buffer, self._tau_scheduer)
          self._max_state = deque(max_len=1)

      def _initiate_players(self, predict_p_v: Callable, buffer: deque, tau_scheduler: TauScheduler) -> None:
          self._max_player.initiate(predict_p_v, buffer, tau_scheduler)
          self._min_player.initiate(predict_p_v, buffer, tau_scheduler)

      def _build_network_graph(self, network: NetworkBaseAGZ) -> tf.Session:
          graph = tf.Graph()
          session = tf.Session(graph=graph, config=self.ConfigProto)
          with graph.as_default():
               self._network = network()
          return session

      def _predict_p_v(self, env: Environment) -> Tuple:
          p, v = self._session.run([self._network.p_predicted, self._network.v_predicted],
                                   feed_dict={self._network.state: env.state.canonical})
          return env.predict(p, v)

      def _read_params(self, hyperparams: Dict) -> None:
          self._tau_scheduler_scheme = hyperparams.get("tau_scheduler_scheme", "exponential")
          self._tau_range = hyperparams.get("tau_range", (0.99, 0.1))
          self._batch_size = hyperparams.get("batch_size", 32)

      def action(self, env: Environment) -> Tuple[str, Any]:
          max_action = self._max_player.action(env)
          env.step(max_action)
          self._max_state.append(self._max_player.state(env)+(env.reward,))
          min_action = None
          if not env.ended:
             min_action = self._min_player.action(env)
          return max_action, min_action

      def state(self, env: Environment) -> np.ndarray:
          min_state, min_path = self._min_player.state(env)
          return (min_state, min_path,)+tuple(self._max_state)

      def train(self) -> float:
          batch_size = min(self._batch_size, len(self._self_play_buffer))
          self_play_data = np.array(sample(self._self_play_buffer, k=batch_size), dtype=np.float32)
          state = np.vstack(self_play_data[:, 0])
          p_target = np.vstack(self_play_data[:, 1])
          v_target = np.vstack(self_play_data[:, 2])
          loss, _ = self._session.run([self._network.loss, self._network.grad], feed_dict={self._network.state: state,
                                                                                           self._network.p_target: p_target,
                                                                                           self._network.v_target: v_target})
          return loss

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
