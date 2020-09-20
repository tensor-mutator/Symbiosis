__all__ = ["Progress"]

class Progress:

      def __init__(self, observe: int = 0, explore: int = 0) -> None:
          self._observe = observe
          self._explore = explore
          self._clock = 0
          self._episodic_clock = 0
          self._episode = 0

      @property
      def clock(self) -> int:
          return self._clock

      @clock.setter
      def clock(self, clock: int) -> None:
          self._clock = clock

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

      def bump_episode(self) -> None:
          self._episode += 1
          self._episodic_clock = 0

      def reset(self) -> None:
          self._clock = 0
          self._episodic_clock = 0
          self._episode = 0
