import tensorflow as tf
from ...agent import Agent
from ....environment import Environment
from .replay import ExperienceReplay
from .prioritized_replay import PrioritizedExperienceReplay
from .network import Network
from ...Utilities.lr_scheduler import LRScheduler

class DDQN(Agent):

      def __init__(self, env: Environment, **hyperparams) -> None:
          self._env = env
          self._alias = 'ddqn'
          self._observe = hyperparams.get('observe', 5000)
          self._explore = hyperparams.get('explore', 10000)
          self._batch_size = hyperparams.get('batch_size', 32)
          self._replay_limit = hyperparams.get('replay_limit', 10000)
          self._epsilon_range = hyperparams.get('epsilon_range', (1, 0.0001))
          self._training_interval = hyperparams.get('training_interval', 5)
          self._target_frequency = hyperparams.get('target_frequency', 3000)
          self._network = hyperparams.get('network', 'dueling')
          self._replay_type = hyperparams.get('replay', 'prioritized')
          self._decay_type = hyperparams.get('decay_type', 'linear')
          self._gamma = hyperparams.get('gamma', 0.9)
          self._alpha = hyperparams.get('alpha', 0.7)
          self._beta = hyperparams.get('beta', 0.5)
          self._replay = ExperienceReplay(self._replay_limit,
                                          self._batch_size) if replay_type == 'regular' else PrioritizedExperienceReplay(self._alpha,
                                                                                                                         self._beta,
                                                                                                                         self._replay_limit,
                                                                                                                         self._batch_size)
          self._lr = hyperparams.get('learning_rate', 0.0001)
          self._lr_scheduler_scheme = hyperparams.get('lr_scheduler_scheme', 'linear')
          self._lr_scheduler = LRScheduler(self._lr_scheduler_scheme, self._lr)
          self._session = self._build_network_graph(hyperparams)
          self._q_update_session = self._build_td_update_graph()
          
      def _build_network_graph(self) -> tf.Session:
          graph = tf.Graph()
          session = tf.Session(graph=graph)
          with graph.as_default():
               network_params = self._get_network_params(self._env.state.size, self._env.action.size, hyperparams)
               self._local_network = getattr(Network, self._network)
               self._target_network = getattr(Network, self._network)
          return session
