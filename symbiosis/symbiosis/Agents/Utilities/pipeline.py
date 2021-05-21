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
          self._y_shapes = list(map(lambda meta: meta["shape"], list(meta_y.values())))
          self._iterator, self._X_fit, self._ys_fit = self._generate_iterator(meta_X, meta_y, batch_size)
          self._fit_model = self._build_fit_graph(self._iterator, model, **params)
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
          placeholders_y = list()
          for id, meta in meta_y.items():
              placeholder_y = tf.placeholder(shape=(None, meta["shape"],), dtype=meta["dtype"])
              placeholders_y.append(placeholder_y)
          dataset = tf.data.Dataset.from_tensor_slices((placeholder_X,)+tuple(placeholders_y))
          dataset = dataset.shuffle(tf.cast(tf.shape(placeholder_X)[0], tf.int64)).batch(batch_size).prefetch(1)
          return dataset.make_initializable_iterator(), placeholder_X, placeholders_y

      def _build_fit_graph(self, iterator: tf.data.Iterator, model: Model, **params) -> Model:
          placeholders = iterator.get_next()
          placeholder_X = placeholders[0]
          placeholders_y = {id: plc for id, plc in zip(self._y_ids, placeholders[1:])}
          with tf.variable_scope("FIT"):
               model = model(placeholder_X=placeholder_X, shapes_y={id: shape for id, shape in zip(self._y_ids, self._y_shapes)},
                             placeholders_y=placeholders_y, **params)
          return model

      def _build_predict_graph(self, meta_X: Dict, model: Model, **params) -> Model:
          with tf.variable_scope("PREDICT"):
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
               return self._session.run(list(self._predict_model.y_hat.values()), feed_dict={self._X_predict: X})

      def fit(self, X_train: np.ndarray, X_test: np.ndarray, ys_train: List[np.ndarray],
              ys_test: List[np.ndarray], n_epochs: int) -> float:
          with self._fit_context():
               self._fit(X_train, X_test, ys_train, ys_test, n_epochs)

      def _fit(self, X_train: np.ndarray, X_test: np.ndarray, ys_train: List[np.ndarray],
               ys_test: List[np.ndarray], n_epochs: int) -> float:
          def feed_dict(X: np.ndarray, ys: np.ndarray) -> Dict:
              feed_dict = {self._X_fit: X}
              for plc, y in zip(self._ys_fit, ys):
                  feed_dict.update({plc: y})
              return feed_dict
          def fetch(train=True) -> float:
              if train:
                 loss, _ = self._session.run([self._fit_model.loss, self._fit_model.grad])
                 return loss
              loss = self._session.run(self._fit_model.loss)
              return loss
          n_batches_train = np.ceil(np.size(X_train, axis=0)/self._batch_size)
          n_batches_test = np.ceil(np.size(X_test, axis=0)/self._batch_size)
          with self._session.graph.as_default():
               self._session.run(tf.global_variables_initializer())
               for epoch in range(n_epochs):
                   self._session.run(self._iterator.initializer, feed_dict=feed_dict(X_train, ys_train))
                   with tqdm(total=len(X_train)) as progress:
                        try:
                           train_losss = 0
                           while True:
                                 train_loss += fetch()
                                 progress.update(self._batch_size)
                        except tf.errors.OutOfRangeError:
                           ...
                   self._session.run(self._iterator.initializer, feed_dict=feed_dict(X_test, ys_test))
                   with tqdm(total=len(X_test)) as progress:
                        try:
                           test_loss
                           while True:
                                 test_loss += fetch(train=False)
                                 progress.update(self._batch_size)
                        except tf.errors.OutOfRangeError:
                           ...
                   self._print_summary(epoch+1, train_loss/n_batches_train, test_loss/n_batches_test)

      def _print_summary(self, epoch: int, train_loss: float, test_loss: float) -> None:
          print(f"{COLORS.UP}\r{COLORS.WIPE}\n{COLORS.WIPE}EPOCH: {COLORS.CYAN}{epoch}{COLORS.DEFAULT}")
          print(f"\n\tTraining set:")
          print(f"\n\t\tLoss: {COLORS.GREEN}{train_loss}{COLORS.DEFAULT}")
          print(f"\n\tTest set:")
          print(f"\n\t\tLoss: {COLORS.MAGENTA}{test_loss/n_batches_test}{COLORS.DEFAULT}")

      def _sync_ops(self) -> tf.group:
          trainable_vars_predict = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="PREDICT")
          trainable_vars_fit = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="FIT")
          sync_ops = list()
          for from_ ,to_ in zip(trainable_vars_fit, trainable_vars_predict):
              sync_ops.append(to_.assign(from_))
          return tf.group(sync_ops)

      def __del__(self) -> None:
          self._session.close()
