import os
import math
import json
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import trange
from tokenizer import get_tokenizer
from embedding_layer import get_embedding_layer
from sequence_regeneration_accuracy import SequenceRegenerationAccuracy
from masked_loss import MaskedLoss
from auto_encoder import AutoEncoder
from auto_encoder_config import Config

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    # tf.config.set_logical_device_configuration(
    #     gpus[0],
    #     [tf.config.LogicalDeviceConfiguration(memory_limit=3400)])
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

case_sentence_csv_folder = Config['case_sentence_csv_folder']
csv_file_list = Config['csv_file_list']

sentences = []
for fileindex in trange(len(csv_file_list)):
  df = pd.read_csv(os.path.join(case_sentence_csv_folder, csv_file_list[fileindex]))
  sentences.extend(df['sentence'].tolist())

df = None
print("Total number of sentences:", len(sentences))

""" shuffle sentences """
seed = 1337
rng = np.random.RandomState(seed)
rng.shuffle(sentences)

""" Prepare train and validation Datasets """
BATCH_SIZE = Config['batch_size']
validation_split = Config['validation_split']
num_validation_sentences = int(validation_split * len(sentences))

train_sentences = sentences[:-num_validation_sentences]
val_sentences = sentences[-num_validation_sentences:]

train_ds = tf.data.Dataset.from_tensor_slices(train_sentences).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices(val_sentences).batch(BATCH_SIZE)

""" Tokenizer """
tokenizer = get_tokenizer(train_ds)

""" Embedding Layer """
embedding_layer = get_embedding_layer(tokenizer)

""" Model Config """
ENC_UNITS = Config['encoder_units']
DEC_UNITS = Config['decoder_units']
SEQUENCE_PRED = Config['recurrent_layer_output_sequence']

use_seq_regen_acc = True
if Config['accuracy_metric'] != 'sequence_regeneration_accuracy':
  use_seq_regen_acc = False

EPOCHS = Config['epochs']

""" Build Model """
auto_encoder = AutoEncoder(
  embedding_layer,
  ENC_UNITS,
  DEC_UNITS,
  tokenizer,
  use_seq_regen_acc=use_seq_regen_acc
)

if use_seq_regen_acc:
  metric = SequenceRegenerationAccuracy()
else:
  metric = tf.keras.metrics.SparseCategoricalAccuracy()

auto_encoder.compile(
    optimizer=tf.optimizers.Adam(),
    loss=MaskedLoss(sequence=SEQUENCE_PRED),
    metrics=[metric],
    run_eagerly=True
)

""" Create checkpoint directory """
now = datetime.now() # current date and time
dt_str = now.strftime("D%Y_%m_%d_T%H_%M_%S")
ckpt_folder = os.path.join(Config['model_folder'], f"{dt_str}\checkpoints")
if not os.path.exists(ckpt_folder):
  os.makedirs(ckpt_folder)

steps_per_epoch = math.ceil(len(train_sentences)/BATCH_SIZE)
ckpt_freq = int(steps_per_epoch/4)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_folder,
    save_weights_only=True,
    monitor='batch_loss',
    save_freq=ckpt_freq
)
auto_encoder.set_train_config(num_epochs=EPOCHS, steps_per_epoch=steps_per_epoch, checkpoint_frequency=ckpt_freq)

auto_encoder.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[cp_callback])

""" Write metrics to json file """
metrics_dict = auto_encoder.get_metrics_dict()
final_metrics = {}
for key, metrics in metrics_dict.items():
  for metric_name, metric_tensor in metrics.items():
    final_metrics['train'][key][metric_name] = float(metric_tensor.numpy())

test_metrics_dict = auto_encoder.get_test_metrics_dict()
if test_metrics_dict != None:
  for key, metrics in test_metrics_dict.items():
    for metric_name, metric_tensor in metrics.items():
      final_metrics['test'][key][metric_name] = float(metric_tensor.numpy())

metrics_json_path = os.path.join(Config['model_folder'], f"{dt_str}\metrics.json")
with open(metrics_json_path, 'w') as json_file:
  json.dump(final_metrics, json_file)

"""Write model config into json file """
config_json_path = os.path.join(Config['model_folder'], f"{dt_str}\model_config.json")
with open(config_json_path, 'w') as json_file:
  json.dump(Config, json_file)