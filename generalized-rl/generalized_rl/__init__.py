__version__ = '0.1.0'

import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_eager_execution()
from .environment import Environment, State, Action
from .Agents.flow_base import Flow
from .Agents.DQN.ddqn import DDQN
from .Agents.DQN.network import DQNNet, DRQNNet, DuelingDQNNet
from .Agents.network_base import NetworkBaseDQN
from .Agents.Utilities.lego import NetBlocks
from .Agents.Utilities.flow import LucasKanadeFlow, GunnerFarnebackFlow
from .config import config
from .colors import COLORS

__all__ = ["Environment", "State", "Action", "DDQN", "NetworkBaseDQN", "DQNNet", "DRQNNet", "DuelingDQNNet",
           "config", "NetBlocks", "Flow", "LucasKanadeFlow", "GunnerFarnebackFlow"]
