from .progress import Progress
from ...environment import Environment
import os

class Inventory:

      def __init__(self, name: str, file_name: str, extension: str,
                   env: Environment, agent: str, progress: Progress) -> None:
          self._name = name
          self._file_name = file_name
          self._extension = extension
          self._env = env
          self._agent = agent
          self._progress = progress
          self._inventory_path = self._make_path()

      def _make_path(self) -> str:
          path = os.path.join(self._env.name, self._agent, self._name)
          os.mkdirs(path)
          return path

      @property
      def path(self) -> path:
          if not os.path.exists(os.path.join(self._inventory_path, "EPISODE {}".format(self._progress.episode))):
             os.mkdir(os.path.join(self._inventory_path, "EPISODE {}".format(self._progress.episode)))
          return os.path.join(self._inventory_path, "EPISODE {}".format(self._progress.episode),
                              "{}.{}.{}".format(self._file_name, self._progress.epi_clock, self._extension))
