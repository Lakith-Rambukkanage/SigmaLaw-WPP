import tensorflow as tf
from shape_checker import ShapeChecker
from encoder import Encoder
from decoder import Decoder
from auto_encoder_config import Config

class AutoEncoder(tf.keras.Model):
  def __init__(self, embedding_matrix, enc_units, dec_units, tokenizer, rnn_type='GRU', enable_eager_execution=False,
               use_nearest_token_embedding=Config['use_nearest_token_embedding'], train_embeddings=False,
               enc_return_seq=False, dec_return_seq=False):
    super(AutoEncoder, self).__init__()
    """ Set pre-defined tokenizer """
    self.tokenizer = tokenizer
    self.vocab_size = tokenizer.vocabulary_size()
    self.enable_eager_execution = enable_eager_execution
    self.use_nearest_token_embedding = use_nearest_token_embedding
    if self.use_nearest_token_embedding:
      self.norm_embedding_matrix_transposed = tf.transpose(
        tf.math.l2_normalize(embedding_matrix, axis=-1),
        perm=[1, 0]
      )
    else:
      self.norm_embedding_matrix_transposed = None

    self.embedding_layer = tf.keras.layers.Embedding(
      embedding_matrix.shape[0],
      embedding_matrix.shape[1],
      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
      trainable=train_embeddings,
    )

    """ Initialize Encoder & Decoder """
    self.encoder = Encoder(self.embedding_layer, enc_units, recurrent_layer_type=rnn_type, return_seq=enc_return_seq)
    self.decoder = Decoder(self.embedding_layer, dec_units, self.vocab_size, recurrent_layer_type=rnn_type, return_seq=dec_return_seq)

    self.shape_checker = ShapeChecker()

  def call(self, inputs, training=False):
    tokens, mask = self._preprocess(inputs)
    max_seq_length = tf.shape(tokens)[1]

    enc_output, enc_state = self.encoder(tokens, training=training)

    dec_state = enc_state
    # pred_seq = tf.constant(tokens[:, :1], dtype=tokens.dtype)

    for t in tf.range(max_seq_length-1):
      dec_input_tokens = tokens[:, t:t+2]
      loss_weights = tf.cast(mask[:, t+1], dtype=tf.float32)
      # step_loss, pred_token, dec_state = self._loop_step(dec_input_tokens, mask, enc_output, dec_state, training_mode=training)
      step_loss, dec_state = self._loop_step(dec_input_tokens, enc_output, dec_state, loss_weights, training_mode=training)

    #   pred_seq = tf.concat([pred_seq, tf.expand_dims(pred_token, -1)], -1)
    # return pred_seq

  def encode(self, inputs, training=False, return_state=False):
    tokens, mask = self._preprocess(inputs)
    enc_output, enc_state = self.encoder(tokens, training=training)
    if return_state:
      return enc_output, enc_state
    else:
      return enc_output

  def train_step(self, inputs):
    self.shape_checker = ShapeChecker()
    # return self._train_step(inputs)
    if self.enable_eager_execution:
      return self._train_step(inputs)
    else:
      return self._tf_train_step(inputs)

  def test_step(self, inputs):
    self.shape_checker = ShapeChecker()
    # return self._test_step(inputs)
    if self.enable_eager_execution:
      return self._test_step(inputs)
    else:
      return self._tf_test_step(inputs)

  def _preprocess(self, text):
    self.shape_checker(text, ('batch',))

    """ Convert the text to token IDs """
    tokens = self.tokenizer(text)
    self.shape_checker(tokens, ('batch', 's'))

    """ Convert IDs to masks. """
    mask = tokens != 0
    self.shape_checker(mask, ('batch', 's'))

    return tokens, mask

  def _train_step(self, input_text):

    tokens, mask = self._preprocess(input_text)
    max_seq_length = tf.shape(tokens)[1]

    with tf.GradientTape() as tape:
      """ Encode the input """
      enc_output, enc_state = self.encoder(tokens)
      self.shape_checker(enc_output, ('batch', 'enc_units'))
      # self.shape_checker(enc_state, ('batch', 'enc_units'))

      dec_state = enc_state
      loss = tf.constant(0.0)
      # if self.use_seq_regen_acc:
      #   pred_seq = tf.constant(tokens[:, :1], dtype=tokens.dtype)

      for t in tf.range(max_seq_length - 1):
        dec_input_tokens = tokens[:, t:t+2]
        loss_weights = tf.cast(mask[:, t+1], dtype=tf.float32)
        step_loss, dec_state = self._loop_step(dec_input_tokens, enc_output, dec_state, loss_weights, training_mode=True)

        loss = loss + step_loss
        # if self.use_seq_regen_acc:
        #   pred_seq = tf.concat([pred_seq, tf.expand_dims(pred_token, -1)], -1)

      """ Average the loss over all non padding tokens. """
      # average_loss = loss / tf.reduce_sum(tf.cast(mask, tf.float32))
      average_loss = loss / tf.cast(tf.shape(tokens)[0], dtype=loss.dtype)

      # if self.use_seq_regen_acc:
      #   self.compiled_metrics.update_state(
      #       tokens[:, 1:], pred_seq[:, 1:], sample_weight=tf.cast(mask[:, 1:], dtype=tokens.dtype))

    """ Apply an optimization step """
    variables = self.trainable_variables
    gradients = tape.gradient(average_loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))

    batch_metrics = self.get_batch_metrics()
    batch_metrics['batch_loss'] = average_loss

    return batch_metrics

  def _loop_step(self, new_tokens, enc_output, dec_state, loss_weights=None, training_mode=False):
    input_tokens, target_tokens = new_tokens[:, 0:-1], new_tokens[:, -1]

    """ Run the decoder one step. """
    logits, dec_state = self.decoder(input_tokens, enc_output, state=dec_state, training=training_mode)
    self.shape_checker(logits, ('batch', 'logits'))
    # self.shape_checker(dec_state, ('batch', 'dec_units'))

    """ `self.loss` returns the total for non-padded tokens """
    # y = target_tokens
    y = self.embedding_layer(target_tokens, training=False)

    if self.use_nearest_token_embedding:
      # LINK: https://stackoverflow.com/questions/37558899/efficiently-finding-closest-word-in-tensorflow-embedding
      norm_logits = tf.math.l2_normalize(logits, axis=1)
      # cosine_similarity = tf.matmul(norm_logits, tf.transpose(self.norm_embedding_matrix, perm=[1, 0]))
      cosine_similarity = tf.matmul(norm_logits, self.norm_embedding_matrix_transposed)
      nearest_tokens = tf.argmax(cosine_similarity, axis=1)
      y_pred = self.embedding_layer(nearest_tokens, training=False)
    else:
      y_pred = logits

    step_loss = self.loss(y, y_pred, sample_weight=loss_weights)
    self.compiled_metrics.update_state(y, y_pred, sample_weight=loss_weights)

    # if not self.use_seq_regen_acc:
    #   self.compiled_metrics.update_state(y, y_pred, tf.cast((y != 0), dtype=tf.int32))
    # pred_token = tf.math.argmax(y_pred, -1)

    return step_loss, dec_state

  @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
  def _tf_train_step(self, inputs):
    return self._train_step(inputs)

  def _test_step(self, data):
    tokens, mask = self._preprocess(data)
    max_seq_length = tf.shape(tokens)[1]

    enc_output, enc_state = self.encoder(tokens, training=False)

    dec_state = enc_state
    loss = tf.constant(0.0)
    # if self.use_seq_regen_acc:
    #   pred_seq = tf.constant(tokens[:, :1], dtype=tokens.dtype)

    for t in tf.range(max_seq_length - 1):
      dec_input_tokens = tokens[:, t:t+2]
      loss_weights = tf.cast(mask[:, t+1], dtype=tf.float32)
      step_loss, dec_state = self._loop_step(dec_input_tokens, enc_output, dec_state, loss_weights, training_mode=False)
      loss = loss + step_loss
      # if self.use_seq_regen_acc:
      #   pred_seq = tf.concat([pred_seq, tf.expand_dims(pred_token, -1)], -1)

    # average_loss = loss / tf.reduce_sum(tf.cast(mask, tf.float32))
    average_loss = loss / tf.cast(tf.shape(tokens)[0], dtype=loss.dtype)

    # if self.use_seq_regen_acc:
    #   self.compiled_metrics.update_state(
    #       tokens[:, 1:], pred_seq[:, 1:], sample_weight=tf.cast(mask[:, 1:], dtype=tokens.dtype))

    batch_metrics = self.get_batch_metrics()
    batch_metrics['batch_loss'] = average_loss

    return batch_metrics

  @tf.function
  def _tf_test_step(self, inputs):
    return self._test_step(inputs)

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
