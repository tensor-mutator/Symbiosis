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

      def __init__(self, flow_skip: int = 1, batch_size: int = 32) -> None:
          self._session = self._load_graph()
          self._X, self._y, self._y_hat, self._grad = self._load_ops()
          self._img_buffer = deque(maxlen=flow_skip)
          self._flow = GunnerFarnebackFlow()
          self._writer = tf.summary.FileWriter("REGRET")
          self._batch_size = batch_size
          self._flow_buffer = deque(maxlen=batch_size)
          self._reward_buffer = deque(maxlen=batch_size)
          self._step = 0
          self._train_clock = 0

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
          y = graph.get_operation_by_name("target/y").outputs[0]
          grad = graph.get_operation_by_name("target/train_1")
          return X, y, y_hat, grad

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
          self._ended = False
          self._step += 1
          self._regret = dict(total=0, no_reward=0, success=0, hit=0)
          return state

      def _decide_reward(self, predicted_label: np.ndarray) -> int:
          label = np.argmax(predicted_label)
          if label == 0:
             return 0
          if label == 1:
             return 1
          return -5

      def _one_hot_y(self, y: np.ndarray) -> np.ndarray:
          one_hot = np.zeros(shape=[np.size(y), 3], dtype=np.float32)
          one_hot[np.arange(np.size(y)), y] = 1
          return one_hot

      def _train(self) -> None:
          self._session.run(self._grad, feed_dict={self._X: np.array(self._flow_buffer),
                                                   self._y: self._one_hot_y(np.array(self._reward_buffer))})
          self._flow_buffer.clear()
          self._reward_buffer.clear()

      def step(self, action: Any) -> Sequence:
          state, r_t, done, info = self._env.step(action)
          self.state.frame = state
          flow = cv2.resize(self._flow.flow_map(self._img_buffer[0], state), (64, 64))
          self._flow_buffer.append(flow)
          self._reward_buffer.append(r_t)
          self._train_clock += 1
          if self._train_clock%self._batch_size==0:
             self._train()
          label = self._session.run(self._y_hat, feed_dict={self._X: np.expand_dims(flow, axis=0)})
          self._reward = self._decide_reward(label)
          if self._reward != r_t:
             self._regret["total"] += 1
             if r_t == -5:
                self._regret["hit"] += 1
             if r_t == 0:
                self._regret["no_reward"] += 1
             if r_t == 1:
                self._regret["success"] += 1
             self._reward = r_t
             self._ended = True
          if self._reward == -5:
             self._ended = True
          if self._ended:
             self._rollout_summary()
          self._img_buffer.append(state)
          return cv2.resize(state, (64, 64,)), self._reward, self._ended, info

      def _rollout_summary(self) -> None:
          summary = tf.Summary()
          summary.value.add(tag="REGRET Statistics/Steps - Total Regret", simple_value=self._regret["total"])
          summary.value.add(tag="REGRET Statistics/Steps - Hit Regret", simple_value=self._regret["hit"])
          summary.value.add(tag="REGRET Statistics/Steps - Zero Regret", simple_value=self._regret["no_reward"])
          summary.value.add(tag="REGRET Statistics/Steps - Success Regret", simple_value=self._regret["success"])
          self._writer.add_summary(summary, self._step)

      def render(self) -> np.ndarray:
          state = self._env.render(mode="rgb_array")
          self.state.frame = state
          self._img_buffer.append(state)
          return cv2.resize(state, (64, 64,))

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
