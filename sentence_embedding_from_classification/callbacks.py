import os
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from utils import tf_cos_sim

class CheckpointSaver(tf.keras.callbacks.Callback):
  def __init__(self, ckpt_folder, ckpt_freq, steps_per_epoch, starting_epoch=0) -> None:
    super(CheckpointSaver, self).__init__()
    self.ckpt_folder = ckpt_folder
    self.ckpt_freq = max(1, ckpt_freq)
    self.steps_per_epoch = steps_per_epoch
    self.epoch_counter = starting_epoch

  def on_train_batch_end(self, batch, logs=None):
    step_counter = batch + 1
    if not(step_counter % self.ckpt_freq) and step_counter != self.steps_per_epoch:
      ckpt_name = f'epoch-{self.epoch_counter}_step-{step_counter}'
      self.model.save_weights(os.path.join(self.ckpt_folder, ckpt_name))

  def on_epoch_end(self, epoch, logs=None):
    self.epoch_counter = epoch + 1
    ckpt_name = f'epoch-{self.epoch_counter}'
    self.model.save_weights(os.path.join(self.ckpt_folder, ckpt_name))

class STSEvalCallback(tf.keras.callbacks.Callback):
  def __init__(self, sts_val, sts_loss_weight=10):
    super(STSEvalCallback, self).__init__()
    self.sts_val_s1 = []
    self.sts_val_s2 = []
    self.sts_val_score = []
    for item in sts_val:
      self.sts_val_s1.append(item['sentence1'])
      self.sts_val_s2.append(item['sentence2'])
      self.sts_val_score.append(item['score'])
    self.sts_val_s1 = tf.constant(self.sts_val_s1, dtype=tf.string)
    self.sts_val_s2 = tf.constant(self.sts_val_s2, dtype=tf.string)
    self.sts_val_score = tf.constant(self.sts_val_score, dtype=tf.float32)
    self.sts_loss_weight = sts_loss_weight

  def on_epoch_end(self, epoch, logs=None):
    vec1 = self.model.encode(self.sts_val_s1)
    vec2 = self.model.encode(self.sts_val_s2)
    pred_similarity = tf_cos_sim(vec1, vec2)
    sts_loss = tf.keras.losses.mean_squared_error(self.sts_val_score, pred_similarity) * self.sts_loss_weight
    pearson_cosine, _ = pearsonr(self.sts_val_score.numpy(), pred_similarity.numpy())
    spearman_cosine, _ = spearmanr(self.sts_val_score.numpy(), pred_similarity.numpy())
    print(f" **** val_sts_loss: {sts_loss}, val_pearson: {pearson_cosine}, val_spearman: {spearman_cosine} **** ")
