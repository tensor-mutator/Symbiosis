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
          os.makedirs(path)
          return path

      @property
      def path(self) -> str:
          if not os.path.exists(os.path.join(self._inventory_path, "EPISODE {}".format(str(self._progress.episode).zfill(10)))):
             os.mkdir(os.path.join(self._inventory_path, "EPISODE {}".format(str(self._progress.episode).zfill(10))))
          return os.path.join(self._inventory_path, "EPISODE {}".format(str(self._progress.episode).zfill(10)),
                              "{}.{}.{}".format(self._file_name, str(self._progress.epi_clock).zfill(10), self._extension))
