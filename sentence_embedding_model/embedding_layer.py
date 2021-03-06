import os
import io
import numpy as np
import tensorflow as tf
# from tensorflow.keras.layers import Embedding
from auto_encoder_config import Config
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastTextKeyedVectors

# Glove Wikipedia + Gigaword: http://nlp.stanford.edu/data/glove.6B.zip
# GoogleNews Word2Vec: https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
# FastText WikiNews: https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

EMBED_DIM = Config['embed_dim']
EMBED_TYPE = Config['word_embeddings_type']
EMBEDDINGS_PATH = Config['pretrained_word_embeddings_path']

def get_glove_embeddings_dict(embeddings_folder=EMBEDDINGS_PATH, embedding_dim=EMBED_DIM):
  # path_to_glove_file = f"/content/glove.6B.{embedding_dim}d.txt"
  path_to_glove_file = os.path.join(embeddings_folder, f"glove.6B.{embedding_dim}d.txt")

  embeddings_dict = {}
  with open(path_to_glove_file, encoding="utf-8") as f:
    for line in f:
      word, coefs = line.split(maxsplit=1)
      coefs = np.fromstring(coefs, "f", sep=" ")
      embeddings_dict[word] = coefs

  print(f"Found {len(embeddings_dict)} Glove word vectors in {path_to_glove_file}.")

  return embeddings_dict

def create_matrix_from_dict(vocab_list, embeddings_dict, embedding_dim=EMBED_DIM):
  # embeddings_dict = get_glove_embeddings_dict()
  num_tokens = len(vocab_list)
  embedding_found = 0

  # Prepare embedding matrix
  embedding_matrix = np.zeros((num_tokens, embedding_dim), dtype=np.float32)
  # for word, i in word_index.items():
  for i, word in enumerate(vocab_list):
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
      # Words not found in embedding index will be all-zeros.
      # This includes the representation for "padding" and "OOV"
      embedding_matrix[i] = embedding_vector
      embedding_found += 1

  print("Converted %d words out of %d" % (embedding_found, num_tokens))
  return embedding_matrix


def get_word2vec_embedding_matrix(vocab_list, embeddings_path=EMBEDDINGS_PATH, embedding_dim=EMBED_DIM):
  w2v_model = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)

  num_tokens = len(vocab_list)
  embedding_found = 0

  # Prepare embedding matrix
  embedding_matrix = np.zeros((num_tokens, embedding_dim), dtype=np.float32)

  for i in range(num_tokens):
    try:
      vector = w2v_model[vocab_list[i]]
      embedding_matrix[i] = vector
      embedding_found += 1
    except KeyError:
      continue

  print("Converted %d words out of %d" % (embedding_found, num_tokens))
  return embedding_matrix


def get_fasttext_embeddings_dict(embeddings_path=EMBEDDINGS_PATH):
  fin = io.open(embeddings_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
  n, d = map(int, fin.readline().split())
  embeddings_dict = {}
  for line in fin:
    word, coefs = line.rstrip().split(maxsplit=1)
    coefs = np.fromstring(coefs, "f", sep=" ")
    embeddings_dict[word] = coefs
  fin.close()
  print(f"Found {len(embeddings_dict)} FastText word vectors in {embeddings_path}.")
  return embeddings_dict

def get_embeddings_matrix(vocab_list, embedding_type=EMBED_TYPE):
  if embedding_type == 'Glove':
    embeddings_dict = get_glove_embeddings_dict()
    return create_matrix_from_dict(vocab_list, embeddings_dict)
  elif embedding_type == 'FastText':
    embeddings_dict = get_fasttext_embeddings_dict()
    return create_matrix_from_dict(vocab_list, embeddings_dict)
  elif embedding_type == 'Word2Vec':
    return get_word2vec_embedding_matrix(vocab_list)
  elif embedding_type in ['Glove_Legal', 'Word2Vec_Legal', 'FastText_Legal']:
    return get_legal_embedding_matrix(embedding_type, vocab_list)
  else:
    raise ValueError(f"Embedding Type should be one of ['Glove', 'Word2Vec', 'FastText'], given {embedding_type}")

def get_legal_embedding_matrix(emb_type, vocab_list, embeddings_path=EMBEDDINGS_PATH, embedding_dim=EMBED_DIM):
  if emb_type in ['Glove_Legal', 'Word2Vec_Legal']:
    keyed_vec = KeyedVectors.load(embeddings_path, mmap='r')
  else:
    keyed_vec = FastTextKeyedVectors.load(embeddings_path)

  num_tokens = len(vocab_list)
  embedding_found = 0

  # Prepare embedding matrix
  embedding_matrix = np.zeros((num_tokens, embedding_dim), dtype=np.float32)

  for i in range(num_tokens):
    try:
      if vocab_list[i] == '[START]': embedding_matrix[i][0] = 1.0
      elif vocab_list[i] == '[END]': embedding_matrix[i][-1] = 1.0
      else:
        vector = keyed_vec[vocab_list[i]]
        embedding_matrix[i] = vector
        embedding_found += 1
    except KeyError:
      continue

  print("Converted %d words out of %d" % (embedding_found, num_tokens))
  return embedding_matrix

"""
def get_embedding_layer(tokenizer, embedding_type=EMBED_TYPE, trainable=False):
  if embedding_type == 'Glove':
    embeddings_dict = get_glove_embeddings_dict()

  embedding_matrix = get_embeddings_matrix(tokenizer, embeddings_dict)

  embedding_layer = Embedding(
    embedding_matrix.shape[0],
    embedding_matrix.shape[1],
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=trainable,
  )
  return embedding_layer
"""
