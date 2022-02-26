import os
import re
import json
from random import Random
import pandas as pd
from tqdm import trange
import tensorflow as tf
from model_config import dataset_config, train_config
from model_config import sentence_classification_config as cls_config

def preprocess(text):
  processed_text = text.replace('\\', '')
  symbols_regex = re.compile(r"([.()[\]{}!?$@&#*/_;,`~:\-<>\+=])")
  processed_text = symbols_regex.sub(" \\1 ", processed_text)
  processed_text = re.sub(r'\s+', ' ', processed_text)
  processed_text = processed_text.replace('[ CITE ]', 'CITET')
  processed_text = processed_text.replace('"', ' " ')
  processed_text = processed_text.replace("'", " ' ")
  processed_text = re.sub(r'(?<=[.])(?=[^\s])', r' ', processed_text)
  processed_text = processed_text.strip()
  # if processed_text.isupper():
  # if processed_text[-1] == '.': processed_text = processed_text[:-1] + " ."
  processed_text = processed_text.lower()
  return re.sub(r'\s+', ' ', processed_text)

def get_sentences():
  csv_file_list = dataset_config['csv_file_list']
  sentences = []
  for fileindex in trange(len(csv_file_list)):
    df = pd.read_csv(os.path.join(dataset_config['case_sentence_csv_folder'], csv_file_list[fileindex]))
    sentences.extend(df['sentence'].tolist())
  return sentences

def get_general_words():
  with open(dataset_config['general_words_file'], 'r') as f:
    general_words = f.read().splitlines()
  return set(general_words)

def get_replacable_indices():
  with open(dataset_config['replacable_indices_file_path'], 'r') as f:
    repl_labels = json.load(f)
  return repl_labels

def annotate_data(sentences, repl_labels):
  original_sentences_filtered = []
  for i in trange(len(sentences)):
    tokens = preprocess(sentences[i]).split()
    token_len = len(tokens)
    if token_len <= cls_config['seq_len']:
      inds = [cls_config['original_token_label']] * cls_config['seq_len']
      original_sentences_filtered.append({
          'tokens': tokens,
          'changed_indices': inds,
          'replacable_indices': repl_labels[str(i)],
          'changed': cls_config['original_sentence_label']
      })
  return original_sentences_filtered

def get_prepared_data(token_classification=False):
  sentences = get_sentences()
  repl_labels = get_replacable_indices()
  labeled_data = annotate_data(sentences, repl_labels)
  general_words = get_general_words()

  changed_sentence_selector = Random(507)
  change_limit = int(cls_config['changed_data_ratio'] * len(labeled_data))
  changed_sentence_selector.shuffle(labeled_data)

  changed_sentences = labeled_data[:change_limit]
  original_sentences = labeled_data[change_limit:]

  repl_word_selector = Random(23)
  general_word_selector = Random(189)

  for i in trange(len(changed_sentences)):
    repl_indices = [i for i, x in enumerate(changed_sentences[i]['replacable_indices']) if x == 1]
    repl_count = len(repl_indices)
    num_replaces = int(repl_count * cls_config['replacing_ratio'])
    inds = repl_word_selector.sample(repl_indices, num_replaces)
    gen_wrds = general_word_selector.sample(general_words, num_replaces)
    for k, tok in enumerate(changed_sentences[i]['tokens']):
      if k in inds:
        changed_sentences[i]['tokens'][k] = gen_wrds.pop(0)
        changed_sentences[i]['changed_indices'][k] = cls_config['changed_token_label']
    changed_sentences[i]['changed'] = cls_config['changed_sentence_label']

  shuffler = Random(15)
  total_dataset = changed_sentences + original_sentences
  shuffler.shuffle(total_dataset)
  total = len(total_dataset)

  train_set_length = int(train_config['train_set_ratio'] * total)
  val_set_length = int(train_config['val_set_ratio'] * total)

  if token_classification:
    train_ds = tf.data.Dataset.from_tensor_slices((
        [' '.join(d['tokens']) for d in total_dataset[:train_set_length]],
        [d['changed_indices'] for d in total_dataset[:train_set_length]],
    )).batch(train_config['batch_size'])

    val_ds = tf.data.Dataset.from_tensor_slices((
        [' '.join(d['tokens']) for d in total_dataset[train_set_length: train_set_length+val_set_length]],
        [d['changed_indices'] for d in total_dataset[train_set_length: train_set_length+val_set_length]],
    )).batch(train_config['batch_size'])

  else:
    train_ds = tf.data.Dataset.from_tensor_slices((
        [' '.join(d['tokens']) for d in total_dataset[:train_set_length]],
        [d['changed'] for d in total_dataset[:train_set_length]],
    )).batch(train_config['batch_size'])

    val_ds = tf.data.Dataset.from_tensor_slices((
        [' '.join(d['tokens']) for d in total_dataset[train_set_length: train_set_length+val_set_length]],
        [d['changed'] for d in total_dataset[train_set_length: train_set_length+val_set_length]],
    )).batch(train_config['batch_size'])

  return train_ds, val_ds
