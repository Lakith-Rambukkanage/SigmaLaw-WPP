import tensorflow as tf
from utils import tf_cos_sim

class SentenceClassifier(tf.keras.Model):
  def __init__(self, seq_len, tokenizer, rnn_units, rnn_type='GRU', embedding_dim=300):
    super(SentenceClassifier, self).__init__()
    self.seq_len = seq_len
    self.tokenizer = tokenizer
    self.vocab_size = tokenizer.vocabulary_size()
    self.rnn_units = rnn_units

    self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, embedding_dim, embeddings_initializer='uniform')

    if rnn_type == 'GRU':
      self.rnn_layer = tf.keras.layers.GRU(self.rnn_units, input_shape=(self.seq_len, embedding_dim))
    else:
      self.rnn_layer = tf.keras.layers.LSTM(self.rnn_units, input_shape=(self.seq_len, embedding_dim))

    self.dense_layer = tf.keras.layers.Dense(1)

  def set_sts_data(self, sts_train, sts_train_freq, steps_per_epoch, sts_loss_weight=5, sts_batch_size=16):
    self.sts_train_len = len(sts_train)
    self.sts_train_s1 = []
    self.sts_train_s2 = []
    self.sts_train_score = []
    for item in sts_train:
      self.sts_train_s1.append(item['sentence1'])
      self.sts_train_s2.append(item['sentence2'])
      self.sts_train_score.append(item['score'])
    self.sts_train_s1 = tf.constant(self.sts_train_s1, dtype=tf.string)
    self.sts_train_s2 = tf.constant(self.sts_train_s2, dtype=tf.string)
    self.sts_train_score = tf.constant(self.sts_train_score, dtype=tf.float32)
    self.sts_batch_size = sts_batch_size
    self.sts_loss_weight = sts_loss_weight

    self.sts_train_freq = sts_train_freq
    self.steps_per_epoch = steps_per_epoch
    self.sts_batch_counter = 0
    self.step_counter = 0

  def call(self, inputs, training=False, return_rnn_output=False):
    tokens = self.tokenizer(inputs)
    vectors = self.embedding_layer(tokens, training=training)
    rnn_output = self.rnn_layer(vectors, training=training)
    cls = self.dense_layer(rnn_output, training=training)
    if return_rnn_output: return cls, rnn_output
    return cls

  def encode(self, inputs, training=False):
    tokens = self.tokenizer(inputs)
    vectors = self.embedding_layer(tokens, training=training)
    return self.rnn_layer(vectors, training=training)

  def train_step(self, inputs):
    text, label = inputs

    """
    start_ind = self.sts_batch_counter * self.sts_batch_size
    end_ind = (self.sts_batch_counter+1) * self.sts_batch_size
    with tf.GradientTape(persistent=True) as tape:
      pred = self(text, training=True)
      class_loss = self.compiled_loss(
          label, pred, 
          sample_weight=None, regularization_losses=self.losses
      )
      _, vec1 = self(self.sts_train_s1[start_ind : end_ind], training=True, return_rnn_output=True)
      _, vec2 = self(self.sts_train_s2[start_ind : end_ind], training=True, return_rnn_output=True)
      pred_similarity = tf_cos_sim(vec1, vec2)
      sts_loss = tf.keras.losses.mean_squared_error(
          self.sts_train_score[start_ind : end_ind],
          pred_similarity) * self.sts_loss_weight

    loss = class_loss + sts_loss
    # loss = class_loss
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    """

    # # =====================================================

    sts_loss = None
    # if self.step_counter % self.sts_train_freq == 0 or self.step_counter == self.steps_per_epoch:
      # print("step_counter:", self.step_counter)
      # print("sts_batch_counter:", self.sts_batch_counter)
    start_ind = self.sts_batch_counter * self.sts_batch_size
    end_ind = (self.sts_batch_counter+1) * self.sts_batch_size

    vec1 = self.encode(self.sts_train_s1[start_ind : end_ind], training=False)
    with tf.GradientTape() as tape_enc1:
      vec2 = self.encode(self.sts_train_s2[start_ind : end_ind], training=True)
      pred_similarity = tf_cos_sim(vec1, vec2)
      sts_loss = tf.keras.losses.mean_squared_error(
          self.sts_train_score[start_ind : end_ind],
          pred_similarity) * self.sts_loss_weight
    vars = self.embedding_layer.trainable_weights + self.rnn_layer.trainable_weights
    self.optimizer.minimize(sts_loss, vars, tape=tape_enc1)

    vec2 = self.encode(self.sts_train_s2[start_ind : end_ind], training=False)
    with tf.GradientTape() as tape_enc2:
      vec1 = self.encode(self.sts_train_s1[start_ind : end_ind], training=True)
      pred_similarity = tf_cos_sim(vec1, vec2)
      sts_loss = tf.keras.losses.mean_squared_error(
          self.sts_train_score[start_ind : end_ind],
          pred_similarity) * self.sts_loss_weight
    self.optimizer.minimize(sts_loss, vars, tape=tape_enc2)

    # rnn_out = self.encode(text, training=False)
    with tf.GradientTape() as tape_cls:
      # pred = self.dense_layer(rnn_out, training=True)
      pred = self(text, training=True)
      class_loss = self.compiled_loss(
          label, pred, 
          sample_weight=None, regularization_losses=self.losses
      )
    self.optimizer.minimize(class_loss, self.trainable_variables, tape=tape_cls)
    # self.optimizer.minimize(class_loss, self.dense_layer.trainable_weights, tape=tape_cls)

    self.compiled_metrics.update_state(label, pred, sample_weight=None)
    batch_metrics = self.get_batch_metrics()
    batch_metrics['class_loss'] = class_loss
    batch_metrics['sts_loss'] = sts_loss if sts_loss != None else tf.constant(-1, dtype=tf.float32)

    # if self.step_counter % self.sts_train_freq == 0:
    self.sts_batch_counter += 1
    if self.sts_batch_counter * self.sts_batch_size >= self.sts_train_len:
      self.sts_batch_counter = 0
    self.step_counter += 1

    if self.step_counter >= self.steps_per_epoch:
      self.step_counter = 0
    
    # print("(End of train_step) step_counter:", self.step_counter)
    # print("(End of train_step) sts_batch_counter:", self.sts_batch_counter)

    return batch_metrics

  def get_batch_metrics(self):
    """ Prepare a dict mapping metric names to current value """
    batch_metrics = {}
    for metric in self.compiled_metrics.metrics:
      result = metric.result()
      if isinstance(result, dict):
        batch_metrics.update(result)
      else:
        batch_metrics[metric.name] = result
    return batch_metrics
