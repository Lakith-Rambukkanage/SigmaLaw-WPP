import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from auto_encoder_config import Config

VOCAB_SIZE = Config['vocab_size']
SEQ_LEN = Config['sequence_length']
VOCAB_FILE = Config['vocab_file']

def tf_preprocess(text):
  text = tf.strings.regex_replace(text, r"[\\]", ' ')
  text = tf.strings.regex_replace(text, r"([.()[\]{}!?$@&#*/_;,`~:\-<>\+=])", " \\1 ")
  text = tf.strings.regex_replace(text, r"\s{2,}", ' ')
  text = tf.strings.regex_replace(text, r'\[ CITE \]', '[CITE]')
  text = tf.strings.regex_replace(text, r"'", " ' ")
  text = tf.strings.regex_replace(text, r'"', ' " ')
  text = tf.strings.regex_replace(text, r"\s{2,}", ' ')
  text = tf.strings.strip(text)
  text = tf.strings.lower(text)
  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text

def get_tokenizer(vocab_file=VOCAB_FILE, seq_len=SEQ_LEN):
  with open(vocab_file, 'r') as f:
    vocab_list = f.read().split('\n')

  vectorizer = TextVectorization(
    max_tokens=len(vocab_list),
    standardize=tf_preprocess,
    output_sequence_length=seq_len,
    vocabulary=vocab_list
  )
  # vectorizer.adapt(train_ds)
  print("Vocabulary Size:", vectorizer.vocabulary_size())
  return vectorizer