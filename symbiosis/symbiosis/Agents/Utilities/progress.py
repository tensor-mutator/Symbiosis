import numpy as np,

__all__ = ["Progress", "ProgressDQN"]

class _Progress:

      def __init__(self, n_steps: int, train_interval: int, explore: float = np.inf) -> None:
          self._n_train_steps = n_steps//train_interval
          self._train_interval = train_interval
          self._explore = explore
          self._clock = 0
          self._train = False
          self._explore_clock = 0
          self._training_clock = 0
          self._episodic_clock = 0
          self._episode = 0      

      @property
      def training_steps(self) -> int:
          return self._n_train_steps

      @property
      def clock(self) -> int:
          return self._clock

      @clock.setter
      def clock(self, clock: int) -> None:
          self._clock = clock

      @property
      def train(self) -> bool:
          return self._train

      @property
      def explore_clock(self) -> int:
          return self._explore_clock

      @property
      def training_clock(self) -> int:
          return self._training_clock

      @property
      def episode(self) -> int:
          return self._episode

      @episode.setter
      def episode(self, episode: int) -> None:
          self._episode = episode

      @property
      def epi_clock(self) -> int:
          return self._episodic_clock

      @epi_clock.setter
      def epi_clock(self, epi_clock: int) -> None:
          self._episodic_clock = epi_clock

      @property
      def explore(self) -> int:
          return self._explore

      def reset(self) -> None:
          self._clock = 0
          self._episodic_clock = 0
          self._episode = 0

      def bump(self) -> None:
          self._clock += 1
          self._episodic_clock += 1
          self._explore_clock = np.clip(self._clock, 0, self._explore)
          self._training_clock = np.clip(self._clock//self._train_interval, 0, self._n_train_steps)
          self._train = True if self._clock%self._train_interval == 0 else False

      def bump_episode(self) -> None:
          self._episode += 1
          self._episodic_clock = 0

class ProgressDQN(_Progress):

      def __init__(self, n_steps: int, train_interval: int, observe: int = 0, 
                   explore: float = np.inf) -> None:
          super().__init__(n_steps, train_interval, explore)
          self._n_train_steps = (n_steps-observe)//train_interval
          self._observe = observe
      
      @property
      def observe(self) -> int:
          return self._observe

      def bump(self) -> None:
          self._clock += 1
          self._episodic_clock += 1
          self._explore_clock = np.clip(self._clock-self._observe, 0, self._explore)
          self._training_clock = np.clip((self._clock-self._observe)//self._train_interval, 0, self._n_train_steps)
          if self._clock > self._observe:
             self._train = True if (self._clock-self._observe)%self._train_interval == 0 else False

class ProgressAGZ(_Progress):

      def __init__(self, n_steps: int, train_interval: int, explore: float = np.inf) -> None:
          super().__init__(n_steps, train_interval, explore)
          self._clock_half = 0
          self._clock_full = 0

      @property
      def clock_half(self) -> int:
          return self._clock

      @property
      def clock_full(self) -> int:
          return self._clock_full

      def bump(self) -> None:
          super().bump()            
          self._clock_full = self._clock//2

class Progress:

      BASE: _Progress = _Progress
      DQN: _Progress = ProgressDQN
      AGZ: _Progress = ProgressAGZ
