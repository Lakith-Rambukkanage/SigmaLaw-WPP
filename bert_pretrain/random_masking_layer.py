import tensorflow as tf

class RandomMasking(tf.keras.layers.Layer):
  def __init__(self, mask_token_id, pad_token_id, start_token_id, end_token_id,
    masks_per_input=0.15, sentence_masking_percentage=0.8) -> None:
    """
    mask_token_id: integer token of [MASK] in the vocabulary
    pad_token_id: integer token of [PAD] (or '') in the vocabulary
    start_token_id: integer token of [CLS] (or [START]) in the vocabulary
    end_token_id: integer token of [SEP] (or [END]) in the vocabulary
    masks_per_input: percentage of tokens to mask in a single example (sentence)
    sentence_masking_percentage: percentage of examples (sentences) to apply masking in the batch
    """
    super(RandomMasking, self).__init__()

    self.mask_token_id = mask_token_id
    self.pad_token_id = pad_token_id
    self.start_token_id = start_token_id
    self.end_token_id = end_token_id
    self.masks_per_input = masks_per_input
    self.sentence_masking_percentage = sentence_masking_percentage

  def call(self, inputs, mask=None):
    """
    Args
      inputs: batch of input tokens for the model (batch_size, seq_len)
      mask: tensor representing whether each token is a subword or not (1 for subword, 0 for not)
    Returns
      masked_tokens: batch of input tokens with randomly applied mask
      mask_loc: tf.int32 tensor of shape (batch_size, seq_len) representing masked tokens (1 for mask, 0 for no mask)
    """
    return self.get_masked_input(inputs, mask)

  def get_masked_input(self, tokens, subword_mask=None):
    """
    Args
      tokens: batch of input tokens for the model (batch_size, seq_len)
      subword_mask: tensor representing whether each token is a subword or not (1 for subword, 0 for not)
    Returns
      masked_tokens: batch of input tokens with randomly applied mask
      mask_loc: tf.int32 tensor of shape (batch_size, seq_len) representing masked tokens (1 for mask, 0 for no mask)
    """
    mask = tf.math.logical_and(tokens != self.start_token_id, tokens != self.end_token_id)
    mask = tf.math.logical_and(mask, tokens != self.pad_token_id)
    if subword_mask is not None:
      mask = tf.math.logical_and(mask, tf.math.logical_not(tf.cast(subword_mask, dtype=tf.bool)))

    mask = tf.math.logical_and(
        mask,
        tf.random.uniform(tokens.shape, maxval=1) < self.masks_per_input
    )
    sentence_filter = tf.random.uniform(tokens.shape[:1], maxval=1) < self.sentence_masking_percentage
    sentence_filter = tf.repeat(tf.expand_dims(sentence_filter, axis=-1), repeats=tokens.shape[1], axis=-1)
    mask = tf.math.logical_and(mask, sentence_filter)
    # mask_id_tensor = tf.constant(self.mask_token_id, dtype=tf.int32, shape=tokens.shape)

    # masked_tokens = tf.where(mask, mask_id_tensor, tokens)
    masked_tokens = tf.where(tf.math.logical_not(mask), tokens, self.mask_token_id)

    return masked_tokens, tf.cast(mask, dtype=tf.int32)