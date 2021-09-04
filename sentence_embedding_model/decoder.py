import tensorflow as tf
from shape_checker import ShapeChecker

class Decoder(tf.keras.layers.Layer):
  def __init__(self, embedding_layer, dec_units, vocab_size, return_seq=False):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.embedding = embedding_layer
    self.vocab_size = vocab_size
    self.return_seq = return_seq

    # The RNN keeps track of what's been generated so far.
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=self.return_seq,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    # This fully connected layer produces the logits for each output token.
    self.fc = tf.keras.layers.Dense(self.vocab_size)

  def call(self, new_tokens, enc_output, state=None, mask=None):
    shape_checker = ShapeChecker()
    shape_checker(new_tokens, ('batch', 't'))
    shape_checker(enc_output, ('batch', 'enc_units'))

    if mask is not None:
      shape_checker(mask, ('batch', 's'))

    if state is not None:
      shape_checker(state, ('batch', 'dec_units'))
    
    # Step 1. Lookup the embeddings
    vectors = self.embedding(new_tokens)
    shape_checker(vectors, ('batch', 't', 'embedding_dim'))

    # Step 2. Process one step with the RNN
    rnn_output, state = self.gru(vectors, initial_state=state)

    if self.return_seq:
      shape_checker(rnn_output, ('batch', 't', 'dec_units'))
    else:
      shape_checker(rnn_output, ('batch', 'dec_units'))
    shape_checker(state, ('batch', 'dec_units'))

    # Step 3. Concatenate the encoder output and RNN output
    concat_output = tf.concat([rnn_output, enc_output], axis=-1)

    # Step 4. Pass through Dense Layer to generate logits for each token in vocab
    dec_output = self.fc(concat_output)
    shape_checker(dec_output, ('batch', 'vocab_size'))

    return dec_output, state
