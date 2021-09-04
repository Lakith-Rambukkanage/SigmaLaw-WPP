import tensorflow as tf

class SequenceRegenerationAccuracy(tf.keras.metrics.Metric):
  def __init__(self, name='seq_regen_acc', **kwargs):
    super(SequenceRegenerationAccuracy, self).__init__(name=name, **kwargs)
    self.regen_acc = self.add_weight(name='regen_acc', dtype=tf.float64, initializer='zeros')
    self.batch_count = self.add_weight(name='batch_count', dtype=tf.int32, initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    """
    y_true: actual token sequence for the batch | (Batch_size, seq_len) | tf.int32
    y_pred: predicted token sequence for the batch | (Batch_size, seq_len) | tf.int32
    sample_weight: mask to identify [PAD] tokens | (Batch_size, seq_len) | tf.int32
    """
    # get boolean tensor for batch prediction
    pred_status = tf.cast(tf.math.equal(y_true, y_pred), tf.int32)

    # count correct predictions
    if sample_weight != None:
      correct_pred_count = tf.math.reduce_sum(tf.math.multiply(pred_status, sample_weight))
      token_count = tf.math.reduce_sum(sample_weight)
    else:
      correct_pred_count = tf.math.reduce_sum(pred_status)
      token_count = tf.constant(pred_status.shape[0] * pred_status.shape[1], dtype=tf.int32)

    self.regen_acc.assign_add(tf.cast(tf.math.divide(correct_pred_count, token_count), dtype=self.regen_acc.dtype))
    self.batch_count.assign_add(1)

  def result(self):
    if self.batch_count == 0:
      return tf.constant(0.0)
    return tf.math.divide(self.regen_acc, tf.cast(self.batch_count, self.regen_acc.dtype))
