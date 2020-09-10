__version__ = '0.1.0'

from .environment import Environment
from .Agents.DQN.ddqn import DDQN

__all__ = ["Environment", "DDQN"]
