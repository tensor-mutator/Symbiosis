import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
from generalized_rl import Environment, State, Action, config
from generalized_rl.Agents import DDQN, DQNNet
from typing import Tuple, Sequence, Any
import numpy as np
import gym
import cv2

class EnduroState(State):

      @property
      def shape(self) -> Tuple[int, int]:
          return (64, 64,)

class EnduroAction(Action):

      @property
      def size(self) -> int:
          return 9

class Enduro(Environment):

      @property
      def name(self) -> str:
          return "Enduro-v0"

      def make(self) -> None:
          self._env = gym.make("Enduro-v0")
          self._ended = False

      def reset(self) -> np.ndarray:
          state = self._env.reset()
          self._state_shape = (state.shape[0], state.shape[1],)
          return state

      def step(self, action: Any) -> Sequence:
          state, self._reward, self._ended, info = self._env.step(action)
          return cv2.resize(state, (64, 64,)), self._reward, self._ended, info

      def render(self) -> np.ndarray:
          return cv2.resize(self._env.render(mode="rgb_array"), (64, 64,))

      @property
      def state(self) -> State:
          return EnduroState()

      @property
      def action(self) -> Action:
          return EnduroAction()

      @property
      def ended(self) -> bool:
          return self._ended
 
      @property
      def reward(self) -> float:
          return self._reward

      def close(self) -> bool:
          self._env.close()

def main():
    agent = DDQN(Enduro(), DQNNet, config=config.VERBOSE_LITE)
    agent.run()
