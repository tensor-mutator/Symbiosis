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
          self._inventory_path = os.path.join(self._env.name, self._agent, self._name)
          if not os.path.exists(self._inventory_path):
             os.makedirs(self._inventory_path)
          return self._inventory_path

      @property
      def inventory_path(self) -> str:
          return self._inventory_path

      @property
      def path(self) -> str:
          if not os.path.exists(os.path.join(self._inventory_path, "EPISODE {}".format(str(self._progress.episode).zfill(10)))):
             os.mkdir(os.path.join(self._inventory_path, "EPISODE {}".format(str(self._progress.episode).zfill(10))))
          extension = "{}.{}.{}".format(self._file_name, str(self._progress.epi_clock).zfill(10), self._extension)
          return os.path.join(self._inventory_path, "EPISODE {}".format(str(self._progress.episode).zfill(10)), extension)

      @property
      def init_path(self) -> str:
          if not os.path.exists(os.path.join(self._inventory_path, "EPISODE {}".format(str(self._progress.episode).zfill(10)))):
             os.mkdir(os.path.join(self._inventory_path, "EPISODE {}".format(str(self._progress.episode).zfill(10))))
          extension = "{}.{}.{}".format(self._file_name, "INIT", self._extension)
          return os.path.join(self._inventory_path, "EPISODE {}".format(str(self._progress.episode).zfill(10)), extension)
