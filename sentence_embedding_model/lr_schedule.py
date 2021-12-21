import math
import tensorflow as tf

class WarmupAndDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """
  Learning rate scheduler described in paper, Attention Is All You Need [arxiv: 1706.03762]
  """
  def __init__(self, d_model, warmup_steps=4000):
    super(WarmupAndDecay, self).__init__()

    # self.d_model = d_model
    self.d_model = tf.constant(d_model, dtype=tf.float32)

    self.scale = warmup_steps ** -1.5

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * self.scale

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class CosineDecayWithStableRange(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, start_lr, end_lr, decay_start_step, total_steps) -> None:
    super(CosineDecayWithStableRange, self).__init__()
    self.start_lr = tf.constant(start_lr, dtype=tf.float64)
    self.end_lr = tf.constant(end_lr, dtype=tf.float64)
    self.decay_start_step = tf.constant(decay_start_step, dtype=tf.float64)
    self.total_steps = tf.constant(total_steps, dtype=tf.float64)
    self.decay_steps = self.total_steps - self.decay_start_step
    self.pi = tf.constant(math.pi, dtype=tf.float64)

  def __call__(self, step):
    arg = tf.math.maximum(step, self.decay_start_step) - self.decay_start_step
    return self.start_lr - (self.start_lr - self.end_lr) * tf.math.sin((self.pi / 2) * (arg / self.decay_steps))
