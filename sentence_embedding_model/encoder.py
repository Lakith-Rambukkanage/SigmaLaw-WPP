import tensorflow as tf
from shape_checker import ShapeChecker

class Encoder(tf.keras.layers.Layer):
  def __init__(self, embedding_layer, enc_units, recurrent_layer_type='GRU', return_seq=False):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.embedding = embedding_layer
    self.recurrent_layer_type = recurrent_layer_type
    self.return_seq = return_seq

    # The embedding layer converts tokens to vectors
    # self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, embedding_dim)

    if self.recurrent_layer_type == 'GRU':
      # The GRU RNN layer processes those vectors sequentially.
      self.recurrent_layer = tf.keras.layers.GRU(
        self.enc_units,
        return_sequences=self.return_seq,
        return_state=True,
        recurrent_initializer='glorot_uniform'
      )

    else:
      self.recurrent_layer = tf.keras.layers.LSTM(self.enc_units, return_sequences=self.return_seq, return_state=True)

  def call(self, tokens, state=None):
    shape_checker = ShapeChecker()
    shape_checker(tokens, ('batch', 's'))

    # 2. The embedding layer looks up the embedding for each token.
    vectors = self.embedding(tokens)
    shape_checker(vectors, ('batch', 's', 'embed_dim'))

    # 3. The RNN Layer processes the embedding sequence.
    #    output shape: (batch, s, enc_units) | (batch, enc_units)
    #    state shape: (batch, enc_units)
    if self.recurrent_layer_type == 'GRU':
      output, state = self.recurrent_layer(vectors, initial_state=state)
    else:
      #    state is a list for LSTM: [state_h, state_c]
      output, state_h, state_c = self.recurrent_layer(vectors, initial_state=state)
      state = [state_h, state_c]

    if self.return_seq:
      shape_checker(output, ('batch', 's', 'enc_units'))
    else:
      shape_checker(output, ('batch', 'enc_units'))
    # shape_checker(state, ('batch', 'enc_units'))

    # 4. Returns the output and state.
    return output, state
