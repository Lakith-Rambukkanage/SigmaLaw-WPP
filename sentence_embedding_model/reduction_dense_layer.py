import tensorflow as tf

class ReductionDense(tf.keras.layers.Layer):
  def __init__(self, seq_len, input_units, include_final_state=False, l1_penalty_ratio=0.01):
    super(ReductionDense, self).__init__()
    self.seq_len = seq_len
    self.input_units = input_units
    self.include_final_state = include_final_state

    self.w1 = self.add_weight(
        shape=(self.seq_len, self.input_units),
        initializer="random_normal",
        trainable=True,
        regularizer=tf.keras.regularizers.L1(l1=l1_penalty_ratio)
    )

    if self.include_final_state:
      self.w2 = self.add_weight(
          shape=(2, self.input_units),
          initializer="random_normal",
          trainable=True
      )

  def call(self, seq_input, mask=None, final_state=None):
    # element-wise multiply w1 & seq_input
    out = tf.math.multiply(seq_input, self.w1)

    # multiply by mask
    if mask != None:
      expanded_mask = tf.cast(tf.expand_dims(mask, -1), self.w1.dtype)
      expanded_mask = tf.repeat(expanded_mask, repeats=self.input_units, axis=-1)
      # expanded_mask = tf.tile(expanded_mask, tf.constant([1, 1, self.input_units]))
      out = tf.math.multiply(out, expanded_mask)
    
    # take the sum along sequence
    out = tf.reduce_sum(out, 1)

    # element-wise multiply w2 & stack(out, final_state)
    if self.include_final_state and final_state != None:
      out = tf.stack([out, final_state], axis=1)
      out = tf.math.multiply(out, self.w2)
      # take the sum along 1st axis
      out = tf.reduce_sum(out, 1)
    
    return out
