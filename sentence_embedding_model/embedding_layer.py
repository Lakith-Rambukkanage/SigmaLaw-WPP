import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from auto_encoder_config import Config

EMBED_DIM = Config['embed_dim']
EMBED_TYPE = Config['word_embeddings_type']
GLOVE_EMBEDDINGS_FOLDER = Config['pretrained_word_embeddings_path']

def get_glove_embeddings_dict(embeddings_folder=GLOVE_EMBEDDINGS_FOLDER, embedding_dim=EMBED_DIM):
  # path_to_glove_file = f"/content/glove.6B.{embedding_dim}d.txt"
  path_to_glove_file = os.path.join(embeddings_folder, f"glove.6B.{embedding_dim}d.txt")

  embeddings_dict = {}
  with open(path_to_glove_file, encoding="utf8") as f:
    for line in f:
      word, coefs = line.split(maxsplit=1)
      coefs = np.fromstring(coefs, "f", sep=" ")
      embeddings_dict[word] = coefs

  print(f"Found {len(embeddings_dict)} Glove word vectors in {path_to_glove_file}.")

  return embeddings_dict

def get_embeddings_matrix(tokenizer, embeddings_dict, embedding_dim=EMBED_DIM):
  num_tokens = tokenizer.vocabulary_size() + 2
  embedding_found = 0

  # Prepare embedding matrix
  embedding_matrix = np.zeros((num_tokens, embedding_dim), dtype=np.float32)
  # for word, i in word_index.items():
  for i, word in enumerate(tokenizer.get_vocabulary()):
    embedding_vector = embeddings_dict.get(word)
    if embedding_vector is not None:
      # Words not found in embedding index will be all-zeros.
      # This includes the representation for "padding" and "OOV"
      embedding_matrix[i] = embedding_vector
      embedding_found += 1

  print("Converted %d words out of %d" % (embedding_found, tokenizer.vocabulary_size()))
  return embedding_matrix

def get_embedding_layer(tokenizer, embedding_type=EMBED_TYPE, trainable=False):
  if embedding_type == 'GLOVE':
    embeddings_dict = get_glove_embeddings_dict()
  
  embedding_matrix = get_embeddings_matrix(tokenizer, embeddings_dict)
  
  embedding_layer = Embedding(
    embedding_matrix.shape[0],
    embedding_matrix.shape[1],
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=trainable,
  )
  return embedding_layer