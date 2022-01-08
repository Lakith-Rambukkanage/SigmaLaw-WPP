import os
import math
import json
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import trange
from tokenizer import get_tokenizer
from embedding_layer import get_embeddings_matrix
# from sequence_regeneration_accuracy import SequenceRegenerationAccuracy
# from masked_loss import MaskedLoss
from custom_callbacks import CheckpointSaver, MetricsRecorder
from auto_encoder import AutoEncoder
from auto_encoder_config import Config, validate_config

train_start_datetime = datetime.now()

validate_config()

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
else:
  print("No GPUs found, executing in CPU")

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

# sentences = sentences[:1000]

""" Prepare train and validation Datasets """
BATCH_SIZE = Config['batch_size']
validation_split = Config['validation_split']
num_validation_sentences = int(validation_split * len(sentences))

train_sentences = sentences[:-num_validation_sentences]
val_sentences = sentences[-num_validation_sentences:]

train_ds = tf.data.Dataset.from_tensor_slices(train_sentences).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices(val_sentences).batch(BATCH_SIZE)

""" Tokenizer """
tokenizer = get_tokenizer()

""" Embedding Layer """
embedding_matrix = get_embeddings_matrix(tokenizer.get_vocabulary())

""" Model Config """
RNN_TYPE = Config['recurrent_layer']
ENC_UNITS = Config['encoder_units']
DEC_UNITS = Config['decoder_units']
SEQUENCE_PRED = Config['recurrent_layer_output_sequence']

# use_seq_regen_acc = True
# if Config['accuracy_metric'] != 'sequence_regeneration_accuracy':
#   use_seq_regen_acc = False

NUM_EPOCHS = Config['num_epochs']
INIT_EPOCH = Config['starting_epoch']

""" Build Model """
auto_encoder = AutoEncoder(
  embedding_matrix,
  ENC_UNITS,
  DEC_UNITS,
  tokenizer,
  rnn_type=RNN_TYPE,
  enable_eager_execution=False
)

ckpt_path = Config['pre_trained_ckpt']

if ckpt_path != None and len(glob.glob(f'{ckpt_path}.*')) == 2:
  auto_encoder.load_weights(ckpt_path)
  print(f"Loaded checkpoint: {ckpt_path}")

# if use_seq_regen_acc:
#   metric = SequenceRegenerationAccuracy()
# else:
#   metric = tf.keras.metrics.SparseCategoricalAccuracy()

auto_encoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=Config['learning_rate']),
    # loss=MaskedLoss(sequence=SEQUENCE_PRED),
    loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
    # metrics=[metric],
    metrics=[tf.keras.metrics.CosineSimilarity()],
    # run_eagerly=True
)

""" Create checkpoint directory """
now = datetime.now() # current date and time
dt_str = now.strftime("D%Y_%m_%d_T%H_%M_%S")
current_model_folder = os.path.join(Config['model_folder'], f'AutoEncoder_{Config["word_embeddings_type"]}', dt_str)
ckpt_folder = os.path.join(current_model_folder, "checkpoints")

if not os.path.exists(ckpt_folder):
  os.makedirs(ckpt_folder)

steps_per_epoch = math.ceil(len(train_sentences)/BATCH_SIZE)
ckpt_freq = int(steps_per_epoch / Config['checkpoints_per_epoch'])

# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=os.path.join(ckpt_folder, 'ckpt-{epoch:02d}'),
#     save_weights_only=True,
#     monitor='batch_loss',
#     save_freq=ckpt_freq
# )
cp_callback = CheckpointSaver(ckpt_folder, ckpt_freq, steps_per_epoch, starting_epoch=INIT_EPOCH)

metrics_json_path = os.path.join(current_model_folder, "metrics.json")
metric_callback = MetricsRecorder(metrics_json_path, int(steps_per_epoch / Config['logs_per_epoch']), steps_per_epoch, starting_epoch=INIT_EPOCH)

auto_encoder.fit(
  train_ds, validation_data=val_ds, epochs=INIT_EPOCH + NUM_EPOCHS, initial_epoch=INIT_EPOCH,
  callbacks=[cp_callback, metric_callback]
)
# auto_encoder.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)


""" Write model config into json file """
config_json_path = os.path.join(current_model_folder, "model_config.json")
with open(config_json_path, 'w') as json_file:
  json.dump(Config, json_file)

end_time = datetime.now()
print(f"============= Total Training Time: {end_time - train_start_datetime} ============")
