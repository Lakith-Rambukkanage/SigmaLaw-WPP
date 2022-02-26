import tensorflow as tf

class ClassWisePrecision(tf.keras.metrics.Precision):
  def __init__(self, class_label, name=None, **kwargs):
    if name == None: name = f'precision_c{class_label}'
    super(ClassWisePrecision, self).__init__(name=name, **kwargs)
    self.class_label = class_label

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred_label = tf.where(y_pred >= 0.0, 1, 0)
    super().update_state(
        tf.where(y_true==self.class_label, 1, 0),
        tf.where(y_pred_label==self.class_label, 1, 0),
        sample_weight
    )

class ClassWiseRecall(tf.keras.metrics.Recall):
  def __init__(self, class_label, name=None, **kwargs):
    if name == None: name = f'recall_c{class_label}'
    super(ClassWiseRecall, self).__init__(name=name, **kwargs)
    self.class_label = class_label

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred_label = tf.where(y_pred >= 0.0, 1, 0)
    super().update_state(
        tf.where(y_true==self.class_label, 1, 0),
        tf.where(y_pred_label==self.class_label, 1, 0),
        sample_weight
    )
