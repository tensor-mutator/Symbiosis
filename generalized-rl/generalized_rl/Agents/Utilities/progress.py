import numpy as np

__all__ = ["Progress"]

class Progress:

      def __init__(self, n_steps: int, observe: int = 0, explore: float = np.inf) -> None:
          self._n_train_steps = n_steps-observe
          self._observe = observe
          self._explore = explore
          self._clock = 0
          self._explore_clock = 0
          self._training_clock = 0
          self._episodic_clock = 0
          self._episode = 0

      @property
      def clock(self) -> int:
          return self._clock

      @clock.setter
      def clock(self, clock: int) -> None:
          self._clock = clock

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
      def observe(self) -> int:
          return self._observe

      @property
      def explore(self) -> int:
          return self._explore

      def bump(self) -> None:
          self._clock += 1
          self._episodic_clock += 1
          self._explore_clock = np.clip(self._clock-self._observe, 0, self._explore)
          self._training_clock = np.clip(self._clock-self._observe, 0, self._n_train_steps)

      def bump_episode(self) -> None:
          self._episode += 1
          self._episodic_clock = 0

      def reset(self) -> None:
          self._clock = 0
          self._episodic_clock = 0
          self._episode = 0
