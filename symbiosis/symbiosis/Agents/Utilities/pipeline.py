import tensorflow.compat.v1 as tf
import numpy as np
from typing import Dict, Callable, List, Tuple, Generator
from tqdm import tqdm
from contextlib import contextmanager
from .metrics import Metrics
from ..model import Model
from ...colors import COLORS

class Pipeline:

      def __init__(self, meta_X: Dict, meta_y: Dict, model: Model, batch_size: int = 32, **params) -> None:
          self._batch_size = batch_size
          self._y_ids = list(meta_y.keys())
          self._y_metrics = dict(map(lambda id, meta: (id, meta.get("metrics", None)), list(meta_y.keys()), list(meta_y.values())))
          self._y_shapes = list(map(lambda meta: meta["shape"], list(meta_y.values())))
          self._iterator, self._X_fit, self._ys_fit = self._generate_iterator(meta_X, meta_y, batch_size)
          self._fit_model, self._metrics = self._build_fit_graph(self._iterator, model, **params)
          self._predict_model, self._X_predict = self._build_predict_graph(meta_X, model, **params)
          self._sync = self._sync_ops()
          self._session = tf.Session(config=self._config())

      @property
      def graph(self) -> tf.Graph:
          return self._session.graph

      @property
      def session(self) -> tf.Session:
          return self._session

      def _generate_iterator(self, meta_X: Dict, meta_y: Dict, batch_size: int) -> Tuple[tf.data.Iterator, tf.placeholder, List[tf.placeholder]]:
          placeholder_X = tf.placeholder(shape=(None,)+meta_X["shape"], dtype=meta_X["dtype"])
          placeholders_y = dict()
          for id, meta in meta_y.items():
              placeholder_y = tf.placeholder(shape=(None, meta["shape"],), dtype=meta["dtype"])
              placeholders_y.update({id: placeholder_y})
          dataset = tf.data.Dataset.from_tensor_slices((placeholder_X,)+tuple(list(placeholders_y.values())))
          dataset = dataset.shuffle(tf.cast(tf.shape(placeholder_X)[0], tf.int64)).batch(batch_size).prefetch(1)
          return dataset.make_initializable_iterator(), placeholder_X, placeholders_y

      def _build_fit_graph(self, iterator: tf.data.Iterator, model: Model, **params) -> Tuple[Model, Dict]:
          placeholders = iterator.get_next()
          placeholder_X = placeholders[0]
          placeholders_y = dict(zip(self._y_ids, placeholders[1:]))
          with tf.variable_scope(model.Scope.FIT):
               model = model(placeholder_X=placeholder_X, shapes_y={id: shape for id, shape in zip(self._y_ids, self._y_shapes)},
                             placeholders_y=placeholders_y, **params)
               metric_ops = dict()
               for id, metrics in self._y_metrics.items():
                   if metrics is not None:
                      metric_ops.update({id: dict(list(map(lambda name, metric: (name, metric(model.y[id],
                                                                                              model.y_hat[id])), metrics.keys(), metrics.values())))})
          return model, metric_ops

      def _build_predict_graph(self, meta_X: Dict, model: Model, **params) -> Model:
          with tf.variable_scope(model.Scope.PREDICT):
               placeholder_X = tf.placeholder(shape=(None,)+meta_X["shape"], dtype=meta_X["dtype"])
               model = model(placeholder_X=placeholder_X, shapes_y={id: shape for id, shape in zip(self._y_ids, self._y_shapes)},
                             placeholders_y=None, **params)
          return model, placeholder_X

      @contextmanager
      def _fit_context(self) -> Generator:
          yield
          self._session.run(self._sync)

      def _config(self) -> tf.ConfigProto:
          config = tf.ConfigProto()
          config.gpu_options.allow_growth = True
          return config

      def predict(self, X: np.ndarray) -> Tuple[np.ndarray]:
          with self._session.graph.as_default():
               ys_hat = self._session.run(list(self._predict_model.y_hat.values()), feed_dict={self._X_predict: np.expand_dims(X, axis=0)})
          return list(map(lambda y_hat: np.squeeze(y_hat), ys_hat))

      def fit(self, X_train: np.ndarray, X_test: np.ndarray, ys_train: List[np.ndarray],
              ys_test: List[np.ndarray], n_epochs: int) -> float:
          with self._fit_context():
               self._fit(X_train, X_test, ys_train, ys_test, n_epochs)

      def _fit(self, X_train: np.ndarray, X_test: np.ndarray, ys_train: List[np.ndarray],
               ys_test: List[np.ndarray], n_epochs: int) -> float:
          def feed_dict(X: np.ndarray, ys: np.ndarray) -> Dict:
              feed_dict = {self._X_fit: X}
              for plc, y in zip(list(self._ys_fit.values()), ys):
                  feed_dict.update({plc: y})
              return feed_dict
          def ravel(metrics: Dict) -> List:
              metric_ops = list()
              for meta in list(metrics.values()):
                  if isinstance(meta, dict):
                     for op in list(meta.values()):
                         metric_ops.append(op)
                  else:
                     metric_ops.append(meta)
              return metric_ops
          def unravel(metrics: List, dst_plc: Dict) -> Dict:
              metrics_ = dict()
              start = 0
              for id, metrics_dst in dst_plc.items():
                  if isinstance(metrics_dst, dict):
                     metrics_.update({id: dict(zip(metrics_dst.keys(), metrics[start: len(metrics_dst)]))})
                     start += len(metrics_dst)
                  else:
                     metrics_.update({id: metrics[start]})
                     start += 1
              return metrics_
          def fetch(scores, train=True) -> float:
              prev_scores = ravel(scores)
              if train:
                 scores = self._session.run(ravel(self._metrics)+[self._fit_model.loss, self._fit_model.grad])
                 cum_scores = list(map(lambda prev_score, score: prev_score+score, prev_scores, scores[:-1]))
                 return unravel(cum_scores, self._metrics)
              scores = self._session.run(ravel(self._metrics)+[self._fit_model.loss])
              cum_scores = list(map(lambda prev_score, score: prev_score+score, prev_scores, scores))
              return unravel(cum_scores, self._metrics)
          n_batches_train = np.ceil(np.size(X_train, axis=0)/self._batch_size)
          n_batches_test = np.ceil(np.size(X_test, axis=0)/self._batch_size)
          with self._session.graph.as_default():
               self._session.run(tf.global_variables_initializer())
               for epoch in range(n_epochs):
                   self._session.run(self._iterator.initializer, feed_dict=feed_dict(X_train, ys_train))
                   with tqdm(total=len(X_train)) as progress:
                        try:
                           train_scores = dict(loss=0)
                           train_scores.update({id: {metric: 0 for metric in list(metrics.keys())} for id, metrics in self._metrics.items()})
                           while True:
                                 train_scores = fetch(train_scores)
                                 progress.update(self._batch_size)
                        except tf.errors.OutOfRangeError:
                           ...
                   self._session.run(self._iterator.initializer, feed_dict=feed_dict(X_test, ys_test))
                   with tqdm(total=len(X_test)) as progress:
                        try:
                           test_scores = dict(loss=0)
                           test_scores.update({id: {metric: 0 for metric in list(metrics.keys())} for id, metrics in self._metrics.items()})
                           while True:
                                 test_scores = fetch(test_scores, train=False)
                                 progress.update(self._batch_size)
                        except tf.errors.OutOfRangeError:
                           ...
                   self._print_summary(epoch+1, train_scores, test_scores, n_batches_train, n_batches_test)

      def _print_summary(self, epoch: int, train_scores: Dict, test_scores: Dict, n_batches_train: int, n_batches_test: int) -> None:
          def pretty_print(scores: Dict, n_batches: int) -> None:
              for id, scores in train_scores.items():
                  if isinstance(scores, dict):
                     print(f"\n\t\t{id}:")
                     for metric, score in scores.items():
                         print(f"\n\t\t\t{metric}: {COLORS.GREEN}{score/n_batches}{COLORS.DEFAULT}")
                  else:
                     print(f"\n\t\\tt{id}: {COLORS.GREEN}{scores/n_batches}{COLORS.DEFAULT}")
          print(f"{COLORS.UP}\r{COLORS.WIPE}\n{COLORS.WIPE}EPOCH: {COLORS.CYAN}{epoch}{COLORS.DEFAULT}")
          print(f"\n\tTraining set:")
          pretty_print(train_scores, n_batches_train)
          print(f"\n\tTest set:")
          pretty_print(test_scores, n_batches_test)

      def _sync_ops(self) -> tf.group:
          trainable_vars_predict = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="PREDICT")
          trainable_vars_fit = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="FIT")
          sync_ops = list()
          for from_ ,to_ in zip(trainable_vars_fit, trainable_vars_predict):
              sync_ops.append(to_.assign(from_))
          return tf.group(sync_ops)

      def __del__(self) -> None:
          self._session.close()
