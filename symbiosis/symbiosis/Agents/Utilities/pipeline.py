import tensorflow.compat.v1 as tf
from typing import Dict, Callable, List
from ..model import Model

class Pipeline:

      def __init__(self, meta_X: Dict, meta_y: Dict, model: Model, batch_size: int, **params) -> None:
          self._batch_size = batch_size
          self._y_ids = list(meta_y.keys())
          self._iterator, self._X_fit, self._ys_fit = self._generate_iterator(meta_X, meta_y, batch_size)
          self._fit_model = self._build_fit_graph(self._iterator, model, **params)
          self._predict_model, self._X_predict = self._build_predict_graph(meta_X, model, **params)
          self._session = tf.Session(config=self._config())

      def _generate_iterator(self, meta_X: Dict, meta_y: Dict, batch_size: int) -> Tuple[tf.data.Iterator, tf.placeholder, List[tf.placeholder]]:
          placeholder_X = tf.placeholder(shape=(None,)+meta_X["shape"], dtype=meta_X["dtype"])
          placeholders_y = list()
          for id, meta in meta_y.items():
              placeholder_y = tf.placeholder(shape=(None,)+meta["shape"], dtype=meta["dtype"])
              placeholders_y.append(placeholder_y)
          dataset = tf.data.Dataset.from_tensor_slices((placeholder_X,)+tuple(placeholders_y))
          dataset = dataset.shuffle(tf.cast(tf.shape(placeholder_X)[0], tf.int64)).batch(batch_size).prefetch(1)
          return dataset.make_initializable_iterator(), placeholder_X, placeholders_y

      def _build_fit_graph(self, iterator: tf.data.Iterator, model: Model, **params) -> Model:
          placeholders = iterator.get_next()
          placeholder_X = placeholders[0]
          placeholders_y = {id: plc for id, plc in zip(self_y_ids, placeholders[1:])}
          with tf.variable_scope("FIT"):
               model = model(placeholder_X=placeholder_X, placeholders_y=placeholders_y, **params)
          return model

      def _build_predict_graph(self, meta_X: Dict, model: Model, **params) -> Model:
          with tf.variable_scope("PREDICT"):
               placeholder_X = tf.placeholder(shape=(None,)+meta_X["shape"], dtype=meta_X["dtype"])
               model = model(placeholder_X=placeholder_X, placeholders_y=None, **params)
          return model, placeholder_X

      def _config(self) -> tf.ConfigProto:
          config = tf.ConfigProto()
          config.gpu_options.allow_growth = True
          return config

      def predict(self, X: np.ndarray) -> Tuple[np.ndarray]:
          with self._session.graph.as_default():
               return self._session.run(self._predict_model.y_hat, feed_dict={self._X_predict: X})

      def fit(self, x: np.ndarray, ys: List[np.ndarray]) -> float:
          

      def __del__(self) -> None:
          self._session.close()
