import tensorflow as tf

def tf_cos_sim(v1, v2):
  t1 = tf.math.l2_normalize(v1, axis=1)
  t2 = tf.math.l2_normalize(v2, axis=1)
  return tf.reduce_sum(tf.multiply(t1, t2), axis=-1)
