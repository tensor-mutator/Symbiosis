import tensorflow.compat.v1 as tf
from symbiosis import Environment, State, Action, config
from symbiosis.Agents import DDQN, DQNNet
from symbiosis.Agents.Utilities import GunnerFarnebackFlow
from typing import Tuple, Sequence, Any, List
import numpy as np
from collections import deque
import os
import gym
import gym_ple
import cv2

MODEL = os.path.join(os.path.split(__file__)[0], "GunnerFarnebackRewardModel")

class FlappyBirdState(State):

      @property
      def shape(self) -> Tuple[int, int]:
          return (64, 64,)

class FlappyBirdAction(Action):

      @property
      def size(self) -> int:
          return 2

class FlappyBird(Environment):

      class RewardModelNotFoundError(Exception):

            def __init__(self, msg: str) -> None:
                super(FlappyBird.RewardModelNotFoundError, self).__init__(msg)

      def __init__(self, flow_skip: int = 1) -> None:
          self._session = self._load_graph()
          self._X, self._y_hat = self._load_ops()
          self._img_buffer = deque(maxlen=flow_skip)
          self._flow = GunnerFarnebackFlow()

      def _load_graph(self) -> tf.Session:
          if not os.path.exists(os.path.join(MODEL, "GunnerFarnebackRewardModel.ckpt.meta")):
             raise FlappyBird.RewardModelNotFoundError("Gunner Farneback reward model not found")
          saver = tf.train.import_meta_graph(os.path.join(MODEL, "GunnerFarnebackRewardModel.ckpt.meta"))
          session = tf.Session(config=self._config)
          saver.restore(session, tf.train.get_checkpoint_state(MODEL).model_checkpoint_path)
          return session

      @property
      def _config(self) -> tf.ConfigProto:
          config = tf.ConfigProto()
          config.gpu_options.allow_growth = True
          return config

      def _load_ops(self) -> List:
          graph = self._session.graph
          X = graph.get_operation_by_name("target/X").outputs[0]
          y_hat = graph.get_operation_by_name("target/y_hat/Softmax").outputs[0]
          return X, y_hat

      @property
      def name(self) -> str:
          return "FlappyBird-v0"

      def make(self) -> None:
          self._env = gym.make("FlappyBird-v0")
          self._ended = False

      def reset(self) -> np.ndarray:
          state = self._env.reset()
          self.state.frame = state
          self._state_shape = (state.shape[0], state.shape[1],)
          return state

      def _decide_reward(self, predicted_label: np.ndarray) -> int:
          label = np.argmax(predicted_label)
          if label == 0:
             return 0
          if label == 1:
             return 1
          return -5

      def step(self, action: Any) -> Sequence:
          state, _, _, info = self._env.step(action)
          self.state.frame = state
          downscaled_img = cv2.resize(state, (64, 64,))
          flow = self._flow.flow_map(self._img_buffer[0], downscaled_img)
          label = self._session.run(self._y_hat, feed_dict={self._X: np.expand_dims(flow, axis=0)})
          self._reward = self._decide_reward(label)
          if self._reward == -5:
             self._ended = True
          self._img_buffer.append(downscaled_img)
          return downscaled_img, self._reward, self._ended, info

      def render(self) -> np.ndarray:
          state = self._env.render(mode="rgb_array")
          self.state.frame = state
          downscaled_img = cv2.resize(state, (64, 64,))
          self._img_buffer.append(downscaled_img)
          return downscaled_img

      @property
      def state(self) -> State:
          return FlappyBirdState()

      @property
      def action(self) -> Action:
          return FlappyBirdAction()

      @property
      def ended(self) -> bool:
          return self._ended

      @property
      def reward(self) -> float:
          return self._reward

      def close(self) -> bool:
          self._env.close()

def main():
    agent = DDQN(FlappyBird(), DQNNet, config=config.VERBOSE_LITE+config.REWARD_EVENT+config.LOAD_WEIGHTS+config.SAVE_WEIGHTS)
    agent.run()
