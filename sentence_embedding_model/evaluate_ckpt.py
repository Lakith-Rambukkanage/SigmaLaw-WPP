import glob
import tensorflow as tf
from tokenizer import get_tokenizer
from embedding_layer import get_embeddings_matrix
from auto_encoder import AutoEncoder
from auto_encoder_config import Config

""" Tokenizer """
tokenizer = get_tokenizer()

""" Embedding Layer """
embedding_matrix = get_embeddings_matrix(tokenizer.get_vocabulary())

""" Model Config """
ENC_UNITS = Config['encoder_units']
DEC_UNITS = Config['decoder_units']

""" Build Model """
auto_encoder = AutoEncoder(
  embedding_matrix,
  ENC_UNITS,
  DEC_UNITS,
  tokenizer,
  enable_eager_execution=False
)

ckpt_path = Config['pre_trained_ckpt']

if ckpt_path != None and len(glob.glob(f'{ckpt_path}.*')) == 2:
  auto_encoder.load_weights(ckpt_path)

auto_encoder.compile(
    optimizer=tf.optimizers.Adam(),
    # loss=MaskedLoss(sequence=SEQUENCE_PRED),
    loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
    # metrics=[metric],
    metrics=[tf.keras.metrics.CosineSimilarity()],
    # run_eagerly=True
)
