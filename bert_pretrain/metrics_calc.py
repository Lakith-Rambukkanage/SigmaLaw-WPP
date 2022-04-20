import torch

def masked_token_accuracy(logits, input_ids, labels, mask_token_id):
  """
  logits: (batch_size, seq_len, vocab_size)
  input_ids: (batch_size, seq_len)
  labels: (batch_size, seq_len)
  """
  pred_cls = torch.argmax(logits, dim=-1)
  result = torch.where(torch.eq(pred_cls, labels), 1.0, 0.0) # pred_cls == labels
  masked_indices = torch.where(input_ids==mask_token_id, 1.0, 0.0)
  # masked_indices = masked_indices.to(torch.float32)
  masked_token_result = result * masked_indices
  return torch.sum(masked_token_result) / torch.sum(masked_indices)
