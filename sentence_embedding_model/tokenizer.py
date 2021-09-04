import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from auto_encoder_config import Config

VOCAB_SIZE = Config['vocab_size']
SEQ_LEN = Config['sequence_length']

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

def get_tokenizer(train_ds, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
  vectorizer = TextVectorization(
    max_tokens=vocab_size,
    standardize=tf_preprocess,
    output_sequence_length=SEQ_LEN
  )
  vectorizer.adapt(train_ds)
  print("Vocabulary Size:", vectorizer.vocabulary_size())
  return vectorizer