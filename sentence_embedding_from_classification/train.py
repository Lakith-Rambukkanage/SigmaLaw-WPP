import os
import json
from datetime import datetime
import tensorflow as tf

from model_config import train_config, sts_config, get_config_dict
from model_config import sentence_classification_config as cls_config
from prepare_data import get_prepared_data
from sentence_classifier import SentenceClassifier
from tokenizer import get_tokenizer
from sts_data import get_sts_data
from callbacks import CheckpointSaver, STSEvalCallback
from metrics import ClassWisePrecision, ClassWiseRecall

train_start_datetime = datetime.now()

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_ds, val_ds = get_prepared_data(token_classification=False)
sts_train, sts_val = get_sts_data()

tokenizer = get_tokenizer()

""" Create checkpoint directory """
now = datetime.now() # current date and time
dt_str = now.strftime("D%Y_%m_%d_T%H_%M_%S")
current_model_folder = os.path.join(train_config['model_folder'], dt_str)
ckpt_folder = os.path.join(current_model_folder, "checkpoints")

if not os.path.exists(ckpt_folder):
  os.makedirs(ckpt_folder)

steps_per_epoch = len(train_ds)
ckpt_freq = int(steps_per_epoch / train_config['checkpoints_per_epoch'])

cp_callback = CheckpointSaver(ckpt_folder, ckpt_freq,
    steps_per_epoch, starting_epoch=train_config['starting_epoch'])

sts_eval_callback = STSEvalCallback(sts_val, sts_loss_weight=sts_config['sts_loss_weight'])

model = SentenceClassifier(
    cls_config['seq_len'], tokenizer, train_config['rnn_units'], 
    train_config['recurrent_layer'], train_config['embed_dim']
)

model.set_sts_data(sts_train, 1, steps_per_epoch, sts_config['sts_loss_weight'], sts_config['sts_batch_size'])

original_sentence_precision = ClassWisePrecision(
    cls_config['original_sentence_label'], name='precision_original_sentence')
changed_sentence_precision = ClassWisePrecision(
    cls_config['changed_sentence_label'], name='precision_changed_sentence')

original_sentence_recall = ClassWiseRecall(
    cls_config['original_sentence_label'], name='recall_original_sentence')
changed_sentence_recall = ClassWiseRecall(
    cls_config['changed_sentence_label'], name='recall_changed_sentence')

model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=train_config['learning_rate']),
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  metrics=[
    tf.keras.metrics.BinaryAccuracy(threshold=0.0),
    original_sentence_precision, changed_sentence_precision,
    original_sentence_recall, changed_sentence_recall
  ]
)

INIT_EPOCH = train_config['starting_epoch']
NUM_EPOCHS = train_config['num_epochs']

model.fit(
  train_ds, validation_data=val_ds, epochs=INIT_EPOCH + NUM_EPOCHS, initial_epoch=INIT_EPOCH,
  callbacks=[sts_eval_callback, cp_callback]
)

""" Write model config into json file """
config_json_path = os.path.join(current_model_folder, "model_config.json")
with open(config_json_path, 'w') as json_file:
  json.dump(get_config_dict(), json_file)

end_time = datetime.now()
print(f"============= Total Training Time: {end_time - train_start_datetime} ============")
