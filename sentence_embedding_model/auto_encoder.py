import tensorflow as tf
from shape_checker import ShapeChecker
from encoder import Encoder
from decoder import Decoder

class AutoEncoder(tf.keras.Model):
  def __init__(self, embedding_layer, enc_units, dec_units, tokenizer, use_seq_regen_acc=True,
               seq_len=128, enc_return_seq=False, dec_return_seq=False):
    super(AutoEncoder, self).__init__()
    # Set pre-defined tokenizer
    self.tokenizer = tokenizer
    self.vocab_size = tokenizer.vocabulary_size()
    self.use_seq_regen_acc = use_seq_regen_acc
    self.seq_len = seq_len
    self.test_metrics_dict = None

    # Initialize Encoder & Decoder
    self.encoder = Encoder(embedding_layer, enc_units, enc_return_seq)
    self.decoder = Decoder(embedding_layer, dec_units, self.vocab_size, dec_return_seq)

    self.shape_checker = ShapeChecker()

  def call(self, inputs, training=False):
    tokens, mask = self._preprocess(inputs)
    max_seq_length = tf.shape(tokens)[1]

    enc_output, enc_state = self.encoder(tokens, training=training)

    dec_state = None
    pred_seq = tf.constant(tokens[:, :1], dtype=tokens.dtype)

    for t in tf.range(max_seq_length-1):
      dec_input_tokens = tokens[:, t:t+2]
      step_loss, pred_token, dec_state = self._loop_step(dec_input_tokens, mask, enc_output, dec_state, training_mode=training)
      pred_seq = tf.concat([pred_seq, tf.expand_dims(pred_token, -1)], -1)
    return pred_seq

  def encode(self, inputs, training=False, return_state=False):
    tokens, mask = self._preprocess(inputs)
    enc_output, enc_state = self.encoder(tokens, training=training)
    if return_state:
      return enc_output, enc_state
    else:
      return enc_output

  def set_train_config(self, num_epochs, steps_per_epoch, checkpoint_frequency):
    self.num_epochs = tf.constant(num_epochs, dtype=tf.int32)
    self.steps_per_epoch = tf.constant(steps_per_epoch, dtype=tf.int32)
    self.ckpt_freq = tf.constant(checkpoint_frequency, dtype=tf.int32)
    self.step_counter = tf.constant(0, dtype=tf.int32)
    self.epoch_counter = tf.constant(0, dtype=tf.int32)
    self.metrics_dict = {}

  def get_metrics_dict(self):
    return self.metrics_dict

  def get_test_metrics_dict(self):
    return self.test_metrics_dict

  def train_step(self, inputs):
    # self.run_eagerly = self.enable_eager_execution
    self.step_counter += 1
    if self.step_counter > self.steps_per_epoch:
      self.step_counter = 0
      self.epoch_counter += 1

    self.shape_checker = ShapeChecker()
    return self._train_step(inputs)
    # if self.enable_eager_execution:
    #   return self._train_step(inputs)
    # else:
    #   return self._tf_train_step(inputs)

  def test_step(self, inputs):
    # self.run_eagerly = self.enable_eager_execution
    self.shape_checker = ShapeChecker()
    return self._test_step(inputs)
    # if self.enable_eager_execution:
    #   return self._test_step(inputs)
    # else:
    #   return self._tf_test_step(inputs)

  def _preprocess(self, text):
    self.shape_checker(text, ('batch',))

    # Convert the text to token IDs
    tokens = self.tokenizer(text)
    self.shape_checker(tokens, ('batch', 's'))

    # Convert IDs to masks.
    mask = tokens != 0
    self.shape_checker(mask, ('batch', 's'))

    return tokens, mask

  def _train_step(self, input_text):
    tokens, mask = self._preprocess(input_text)
    # max_seq_length = tf.shape(tokens)[1]

    with tf.GradientTape() as tape:
      # Encode the input
      enc_output, enc_state = self.encoder(tokens)
      self.shape_checker(enc_output, ('batch', 'enc_units'))
      self.shape_checker(enc_state, ('batch', 'enc_units'))

      dec_state = None
      loss = tf.constant(0.0)
      if self.use_seq_regen_acc:
        pred_seq = tf.constant(tokens[:, :1], dtype=tokens.dtype)

      for t in tf.range(tf.constant(self.seq_len-1)):
        dec_input_tokens = tokens[:, t:t+2]
        step_loss, pred_token, dec_state = self._loop_step(dec_input_tokens, mask, enc_output, dec_state, training_mode=True)
        # print("pred_token:", pred_token.shape)
        loss = loss + step_loss
        if self.use_seq_regen_acc:
          pred_seq = tf.concat([pred_seq, tf.expand_dims(pred_token, -1)], -1)

      # Average the loss over all non padding tokens.
      average_loss = loss / tf.reduce_sum(tf.cast(mask, tf.float32))
      # print("tokens:", tokens.shape)
      # print("pred_seq:", pred_seq.shape)
      # print("mask:", mask.shape)

      if self.use_seq_regen_acc:
        self.compiled_metrics.update_state(
            tokens[:, 1:], pred_seq[:, 1:], sample_weight=tf.cast(mask[:, 1:], dtype=tokens.dtype))

    # Apply an optimization step
    variables = self.trainable_variables
    gradients = tape.gradient(average_loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))

    # Prepare a dict mapping metric names to current value
    batch_metrics = self.get_batch_metrics()
    batch_metrics['batch_loss'] = average_loss

    if tf.math.floormod(self.step_counter, self.ckpt_freq) == 0:
      key = f'epoch_{self.epoch_counter}_step_{self.step_counter}'
      self.metrics_dict[key] = batch_metrics

    return batch_metrics

  def _loop_step(self, new_tokens, input_mask, enc_output, dec_state, training_mode=False):
    input_tokens, target_token = new_tokens[:, 0:-1], new_tokens[:, -1]

    # Run the decoder one step.
    logits, dec_state = self.decoder(input_tokens, enc_output, state=dec_state, mask=input_mask, training=training_mode)
    # self.shape_checker(logits, ('batch', 't1', 'logits'))
    self.shape_checker(logits, ('batch', 'logits'))
    self.shape_checker(dec_state, ('batch', 'dec_units'))

    # `self.loss` returns the total for non-padded tokens
    y = target_token
    y_pred = logits
    step_loss = self.loss(y, y_pred)
    if not self.use_seq_regen_acc:
      self.compiled_metrics.update_state(y, y_pred, tf.cast((y != 0), dtype=tf.int32))
    pred_token = tf.math.argmax(y_pred, -1)

    return step_loss, pred_token, dec_state

  # @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
  # def _tf_train_step(self, inputs):
  #   return self._train_step(inputs)

  def _test_step(self, data):
    # self.run_eagerly = self.enable_eager_execution
    tokens, mask = self._preprocess(data)
    # max_seq_length = tf.shape(tokens)[1]

    enc_output, enc_state = self.encoder(tokens, training=False)

    dec_state = None
    loss = tf.constant(0.0)
    if self.use_seq_regen_acc:
      pred_seq = tf.constant(tokens[:, :1], dtype=tokens.dtype)

    for t in tf.range(tf.constant(self.seq_len-1)):
      dec_input_tokens = tokens[:, t:t+2]
      step_loss, pred_token, dec_state = self._loop_step(dec_input_tokens, mask, enc_output, dec_state, training_mode=False)
      loss = loss + step_loss
      if self.use_seq_regen_acc:
        pred_seq = tf.concat([pred_seq, tf.expand_dims(pred_token, -1)], -1)

    # Average the loss over all non padding tokens.
    average_loss = loss / tf.reduce_sum(tf.cast(mask, tf.float32))

    if self.use_seq_regen_acc:
      self.compiled_metrics.update_state(
          tokens[:, 1:], pred_seq[:, 1:], sample_weight=tf.cast(mask[:, 1:], dtype=tokens.dtype))

    # Prepare a dict mapping metric names to current value
    batch_metrics = self.get_batch_metrics()
    batch_metrics['batch_loss'] = average_loss

    if self.test_metrics_dict == None:
      self.test_metrics_dict = {}
    self.test_metrics_dict[f'epoch_{self.epoch_counter}'] = batch_metrics

    return batch_metrics

  # @tf.function
  # def _tf_test_step(self, inputs):
  #   return self._test_step(inputs)

  def get_batch_metrics(self):
    # Prepare a dict mapping metric names to current value
    batch_metrics = {}
    for metric in self.compiled_metrics.metrics:
      result = metric.result()
      if isinstance(result, dict):
        batch_metrics.update(result)
      else:
        batch_metrics[metric.name] = result
    return batch_metrics
