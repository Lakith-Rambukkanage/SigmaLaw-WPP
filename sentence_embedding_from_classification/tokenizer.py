from model_config import dataset_config
from model_config import sentence_classification_config as cls_config
from tensorflow.keras.layers import TextVectorization

def get_vocab():
  with open(dataset_config['vocab_file'], 'r') as f:
    vocab_list = f.read().split('\n')
  vocab_list[vocab_list.index('[cite]')] = 'citet'
  return vocab_list

def get_tokenizer():
  vocab = get_vocab()
  return TextVectorization(
      max_tokens=None, standardize=None, split='whitespace',
      output_sequence_length=cls_config['seq_len'], vocabulary=vocab
  )
