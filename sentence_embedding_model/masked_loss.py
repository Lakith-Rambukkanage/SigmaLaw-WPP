import tensorflow as tf
from shape_checker import ShapeChecker

class MaskedLoss(tf.keras.losses.Loss):
  def __init__(self, sequence=False):
    self.name = 'masked_loss'
    self.sequence = sequence
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  def __call__(self, y_true, y_pred):
    shape_checker = ShapeChecker()
    if self.sequence:
      shape_checker(y_true, ('batch', 't'))
      shape_checker(y_pred, ('batch', 't', 'logits'))
    else:
      shape_checker(y_true, ('batch',))
      shape_checker(y_pred, ('batch', 'logits'))

    # Calculate the loss for each item in the batch.
    loss = self.loss(y_true, y_pred)
    if self.sequence:
      shape_checker(loss, ('batch', 't'))
    else:
      shape_checker(loss, ('batch',))

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, tf.float32)
    if self.sequence:
      shape_checker(mask, ('batch', 't'))
    else:
      shape_checker(mask, ('batch',))

    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)
