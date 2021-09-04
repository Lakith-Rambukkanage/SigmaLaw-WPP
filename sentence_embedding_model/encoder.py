import tensorflow as tf
from shape_checker import ShapeChecker

class Encoder(tf.keras.layers.Layer):
  def __init__(self, embedding_layer, enc_units, return_seq=False):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.embedding = embedding_layer
    self.return_seq = return_seq

    # The embedding layer converts tokens to vectors
    # self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, embedding_dim)

    # The GRU RNN layer processes those vectors sequentially.
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   # Return the sequence and state
                                   return_sequences=self.return_seq,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, tokens, state=None):
    shape_checker = ShapeChecker()
    shape_checker(tokens, ('batch', 's'))

    # 2. The embedding layer looks up the embedding for each token.
    vectors = self.embedding(tokens)
    shape_checker(vectors, ('batch', 's', 'embed_dim'))

    # 3. The GRU processes the embedding sequence.
    #    output shape: (batch, s, enc_units) | (batch, enc_units)
    #    state shape: (batch, enc_units)
    output, state = self.gru(vectors, initial_state=state)
    if self.return_seq:
      shape_checker(output, ('batch', 's', 'enc_units'))
    else:
      shape_checker(output, ('batch', 'enc_units'))
    shape_checker(state, ('batch', 'enc_units'))

    # 4. Returns the output and state.
    return output, state
