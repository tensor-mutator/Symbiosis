import tensorflow.compat.v1 as tf

class Metrics:

      @staticmethod
      def MicroPrecision(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
          y_pred = tf.argmax(y_hat, axis=-1)
          y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
          TP_plus_FP = tf.reduce_sum(y_cap)
          TP = tf.reduce_sum(tf.where(tf.cast(y_cap, tf.bool), y, tf.zeros_like(y_cap)))
          return TP/TP_plus_FP

      @staticmethod
      def MicroRecall(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
          y_pred = tf.argmax(y_hat, axis=-1)
          y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
          TP_plus_FN = tf.reduce_sum(y)
          TP = tf.reduce_sum(tf.where(tf.cast(y, tf.bool), y_cap, tf.zeros_like(y)))
          return TP/TP_plus_FN

      @staticmethod
      def MicroF1Score(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
          MicroPR = MicroPrecision(y, y_hat)
          MicroRC = MicroRecall(y, y_hat)
          return 2*MicroPR*MicroRC/(MicroPR+MicroRC)

      @staticmethod
      def MacroPrecision(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
          y_pred = tf.argmax(y_hat, axis=-1)
          y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
          TP_plus_FP = tf.reduce_sum(y_cap, axis=0)
          TP = tf.reduce_sum(tf.where(tf.cast(y_cap, tf.bool), y, tf.zeros_like(y_cap)), axis=0)
          return tf.reduce_mean(TP/TP_plus_FP)

      @staticmethod
      def MacroRecall(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
          y_pred = tf.argmax(y_hat, axis=-1)
          y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
          TP_plus_FN = tf.reduce_sum(y, axis=0)
          TP = tf.reduce_sum(tf.where(tf.cast(y, tf.bool), y_cap, tf.zeros_like(y)), axis=0)
          return tf.reduce_mean(TP/TP_plus_FN)

      @staticmethod
      def MacroF1Score(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
          MacroPR = MacroPrecision(y, y_hat)
          MacroRC = MacroRecall(y, y_hat)
          return 2*MacroPR*MacroRC/(MacroPR+MacroRC)

      @staticmethod
      def HammingLoss(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
          y_pred = tf.argmax(y_hat, axis=-1)
          y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
          return tf.reduce_sum(tf.cast(tf.not_equal(y, y_cap), tf.int32))/tf.reduce_sum(tf.ones_like(y_cap))
