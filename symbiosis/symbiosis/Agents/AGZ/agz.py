"""
@author: Debajyoti Raychaudhuri

A precise implementation of an AlphaGo Zero agent
"""

import tensorflow.compat.v1 as tf
from typing import Dict, Any, Tuple, Callable
from collections import deque
import numpy as np
import dill
from random import sample
from sklearn.model_selection import train_test_split
from .network import AGZChessNet
from ..agent import AgentMCTS, AgentForked
from ..flow_base import Flow
from ..network_base import NetworkBaseAGZ
from ..Utilities import Progress, TauScheduler
from ...environment import Environment
from ...config import config

@AgentMCTS.track(AGZChessNet)
class AGZ(AgentMCTS):

      def __init__(self, max_min_players: Tuple[AgentForked, AgentForked], env: Environment,
                   network: NetworkBaseAGZ = AGZChessNet, config: bin = config.DEFAULT, flow: Flow = None,
                   **hyperparams) -> None:
          self._max_player, self._min_player = max_min_players
          self._env = env
          self._config = config
          self._flow = flow
          self._alias = network.type
          self._progress = self.load_progress(Progress.AGZ, n_steps=self.total_steps, explore=self.explore)
          self._model = self._build_network_graph(network, hyperparams)
          self._read_params(hyperparams)
          self._tau_scheduer = TauScheduler(self._tau_scheduler_scheme, self._tau_range, self._progress, self._config,
                                            self.writer)
          self._self_play_buffer = deque()
          self._initiate_players(self._predict_p_v, self._self_play_buffer, self._tau_scheduer)
          self._max_state = deque(max_len=1)

      def _initiate_players(self, predict_p_v: Callable, buffer: deque, tau_scheduler: TauScheduler) -> None:
          self._max_player.initiate(predict_p_v, buffer, tau_scheduler, self._progress)
          self._min_player.initiate(predict_p_v, buffer, tau_scheduler, self._progress)

      def _build_network_graph(self, network: NetworkBaseAGZ, hyperparams: Dict) -> tf.Session:
          return network(state_shape=self._env.state.shape, action_size=self._env.action.size, **hyperparams)

      def _predict_p_v(self, env: Environment) -> Tuple:
          p, v = self._model.predict(env.state.canonical)
          return env.predict(p, v)

      def _read_params(self, hyperparams: Dict) -> None:
          self._tau_scheduler_scheme = hyperparams.get("tau_scheduler_scheme", "exponential")
          self._tau_range = hyperparams.get("tau_range", (0.99, 0.1))
          self._n_epochs = hyperparams.get("n_epochs", 1)
          self._explore = hyperparams.get("explore", 50000)
          self._total_steps = hyperparams.get("total_steps", 10000000)

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

      def train(self) -> None:
          self_play_data = np.array(self._self_play_buffer, dtype=np.float32)
          state = np.vstack(self_play_data[:, 0])
          p_target = np.vstack(self_play_data[:, 1])
          v_target = np.vstack(self_play_data[:, 2])
          state_train, state_test, p_target_train, p_target_test, v_target_train, v_target_test = train_test_split([state, p_target, v_target],
                                                                                                                   train_size=0.8, test_size=0.2)
          self._model.fit(X_train=state_train, X_test=state_test, ys_train=[p_target_train, v_target_train],
                                    ys_test=[p_target_test, v_target_test], n_epochs=self._n_epochs)

      @AgentMCTS.register("suite_agz")
      def run(self) -> None:
          ...

      def save(self) -> None:
          self_play_data_path = os.path.join(self.workspace, f"{self._alias}.play")
          with open(self_play_data_path, "wb") as io:
               dill.dump(self._self_play_buffer, io, protocol=dill.HIGHEST_PROTOCOL)

      def load(self) -> None:
          self_play_data_path = os.path.join(self.workspace, f"{self._alias}.play")
          if len(glob(self_play_data_path)) == 0:
             raise MissingDataError("Self play data not found")
          with open(self_play_data_path, "rb") as io:
               self._self_play_buffer = dill.load(io)
