import os
from random import Random
import pandas as pd
import torch
from tqdm import tqdm, trange
from transformers import RobertaTokenizer

from pretrain_config import Config

torch.manual_seed(647)

SEQ_LEN = Config['seq_len']
BATCH_SIZE = Config['batch_size']

class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    # store encodings internally
    self.encodings = encodings

  def __len__(self):
    # return the number of samples
    return self.encodings['input_ids'].shape[0]

  def __getitem__(self, i):
    # return dictionary of input_ids, attention_mask, and labels for index i
    return {key: tensor[i] for key, tensor in self.encodings.items()}

def read_sentences():
  csv_file_list = Config['csv_file_list']
  sentences = []
  for fileindex in trange(len(csv_file_list)):
    df = pd.read_csv(os.path.join(Config['case_sentence_csv_folder'], csv_file_list[fileindex]))
    sentences.extend(df['sentence'].tolist())
  return sentences

def get_tokenizer():
  return RobertaTokenizer.from_pretrained(Config['tokenizer_path'], max_len=SEQ_LEN)

def create_input_ids_masks_labels(sentences, tokenizer):
  input_ids = []
  masks = []
  labels = []
  for sent in tqdm(sentences, total=len(sentences)):
    tokens = tokenizer(sent.lower(), max_length=SEQ_LEN, padding='max_length', truncation=True)
    labels.append(tokens['input_ids'])
    masks.append(tokens['attention_mask'])
    ids = torch.tensor(tokens['input_ids'].copy())
    rand = torch.rand(ids.shape)
    # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
    mask_arr = (rand < .15) * (ids != tokenizer.pad_token_id) * (ids != tokenizer.cls_token_id) * (ids != tokenizer.sep_token_id)
    # get indices of mask positions from mask array
    selection = torch.flatten(mask_arr.nonzero()).tolist()
    # mask input_ids
    ids[selection] = tokenizer.mask_token_id
    input_ids.append(ids)
    # input_ids.append(torch.where(mask_arr, tokenizer.mask_token_id, ids))

  input_ids = torch.stack(input_ids, dim=0)
  masks = torch.tensor(masks)
  labels = torch.tensor(labels)

  return {'input_ids': input_ids, 'attention_mask': masks, 'labels': labels}

def get_data_loader():
  sentences = read_sentences()
  shuffler = Random(15)
  shuffler.shuffle(sentences)
  train_lim = int(len(sentences) * Config['train_set_ratio'])
  train_sentences = sentences[:train_lim]
  val_sentences = sentences[train_lim:]
  tokenizer = get_tokenizer()
  train_encodings = create_input_ids_masks_labels(train_sentences, tokenizer)
  val_encodings = create_input_ids_masks_labels(val_sentences, tokenizer)
  train_dataloader = torch.utils.data.DataLoader(Dataset(train_encodings), batch_size=BATCH_SIZE, shuffle=False)
  val_dataloader = torch.utils.data.DataLoader(Dataset(val_encodings), batch_size=BATCH_SIZE, shuffle=False)
  return train_dataloader, val_dataloader
