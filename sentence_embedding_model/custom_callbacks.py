import os
import json
import tensorflow as tf

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

  # def on_epoch_begin(self, epoch):
  #   self.epoch_counter = epoch

  def on_epoch_end(self, epoch, logs=None):
    self.epoch_counter = epoch + 1
    ckpt_name = f'epoch-{self.epoch_counter}'
    self.model.save_weights(os.path.join(self.ckpt_folder, ckpt_name))

class MetricsRecorder(tf.keras.callbacks.Callback):
  def __init__(self, metrics_json_path, recording_interval, steps_per_epoch, starting_epoch=0) -> None:
    super(MetricsRecorder, self).__init__()
    self.metrics_json_path = metrics_json_path
    self.recording_interval = max(1, recording_interval)
    self.steps_per_epoch = steps_per_epoch
    self.metrics = {}
    self.epoch_counter = starting_epoch

  def on_train_batch_end(self, batch, logs=None):
    step_counter = batch + 1
    if not(step_counter % self.recording_interval) and step_counter != self.steps_per_epoch:
      self.metrics[f'epoch-{self.epoch_counter}_step-{step_counter}'] = logs

  def on_epoch_end(self, epoch, logs=None):
    self.epoch_counter = epoch + 1
    self.metrics[f'epoch-{self.epoch_counter}'] = logs

  def on_train_end(self, logs=None):
    with open(self.metrics_json_path, 'w') as json_file:
      json.dump(self.metrics, json_file)
