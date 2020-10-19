import tensorflow.compat.v1 as tf
from typing import Callable
from .progress import Progress

class RegisterWriter:

      @staticmethod
      def registerwriter(func) -> Callable:
          def inner(inst: "<Scheduler inst>") -> float:
              val = func(inst)
              summary = tf.Summary()
              summary.value.add(tag=inst._tag, simple_value=val)
              inst._writer.add_summary(summary, getattr(inst._progress, inst._y))
              return val
          return inner

class EventWriter(RegisterWriter):

      def _set_writer(self, tag: str, write: bool, writer: tf.summary.FileWriter, progress: Progress,
                      y: str) -> None:
          self._tag = tag
          self._write = write
          self._writer = writer
          self._progress = progress
          self._y = y
