import tensorflow.compat.v1 as tf

class EventWriter:

      def __init__(self, tag: str, write: bool, writer: tf.summary.FileWriter) -> None:
          self._tag = tag
          self._write = write
          self._writer = writer

      def write(self, x: int, y: float) -> None:
          summary = tf.Summary()
          summary.value.add(tag=self_tag, simple_value=y)
          self._writer.add_summary(summary, x)
