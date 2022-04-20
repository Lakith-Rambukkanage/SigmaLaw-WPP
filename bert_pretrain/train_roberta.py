import os
from datetime import datetime
import json
from tqdm import tqdm
import torch
from transformers import RobertaConfig, RobertaForMaskedLM

from pretrain_config import Config
from prepare_inputs import get_data_loader, get_mask_token_id
from metrics_calc import masked_token_accuracy

train_start_datetime = datetime.now()

train_dataloader, val_dataloader = get_data_loader()

mask_token_id = get_mask_token_id()

config = RobertaConfig(
    vocab_size=Config['vocab_size'],  # we align this to the tokenizer vocab_size
    max_position_embeddings=Config['seq_len'],
    hidden_size=Config['embedding_dim'],
    num_attention_heads=Config['attention_heads'],
    num_hidden_layers=Config['encoder_layers'],
    intermediate_size=Config['intermediate_size'],
    type_vocab_size=Config['type_vocab_size']
)

if Config['last_ckpt_path'] == None or Config['last_ckpt_path'] == '':
  model = RobertaForMaskedLM(config)
else:
  model = RobertaForMaskedLM.from_pretrained(Config['last_ckpt_path'], config=config)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)

model.train() # activate training mode
optim = torch.optim.AdamW(model.parameters(), lr=Config['learning_rate'])

dt_str = train_start_datetime.strftime("D%Y_%m_%d_T%H_%M_%S")
model_folder = os.path.join(Config['model_path'], dt_str)
ckpt_path = os.path.join(model_folder, 'checkpoints')
config_path = os.path.join(model_folder, 'train_config.json')
results_path = os.path.join(model_folder, 'results.json')

start_epoch = Config['start_epoch']
end_epoch = start_epoch + Config['num_epochs']
results = {}

for epoch in range(start_epoch, end_epoch):
  # setup loop with TQDM and dataloader
  loop = tqdm(train_dataloader, leave=True)
  total_train_loss = 0.0
  total_train_mask_acc = 0.0

  for batch in loop:
    # initialize calculated gradients (from prev step)
    optim.zero_grad()
    # pull all tensor batches required for training
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    # process
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    mask_acc = masked_token_accuracy(outputs.logits, batch['input_ids'], batch['labels'], mask_token_id)
    total_train_mask_acc += mask_acc.item()
    # extract loss
    loss = outputs.loss
    # calculate loss for every parameter that needs grad update
    loss.backward()
    # update parameters
    optim.step()

    total_train_loss += loss.item()
    # print relevant info to progress bar
    loop.set_description(f'Epoch {epoch+1}')
    loop.set_postfix({'loss': loss.item(), 'mask_acc': mask_acc.item()})

  avg_train_loss = total_train_loss / len(train_dataloader)
  avg_train_mask_acc = total_train_mask_acc / len(train_dataloader)

  total_val_loss = 0.0
  total_val_mask_acc = 0.0
  val_loop = tqdm(val_dataloader, leave=True)

  for batch in val_loop:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    val_loss = outputs.loss
    total_val_loss += val_loss.item()
    mask_acc = masked_token_accuracy(outputs.logits, batch['input_ids'], batch['labels'], mask_token_id)
    total_val_mask_acc += mask_acc.item()
    val_loop.set_description(f'[Val] Epoch {epoch+1}')
    val_loop.set_postfix({'val_loss': val_loss.item(), 'val_mask_acc': mask_acc.item()})

  avg_val_loss = total_val_loss / len(val_dataloader)
  avg_val_mask_acc = total_val_mask_acc / len(val_dataloader)
  print(f'\nloss: {avg_train_loss} | mask_acc: {avg_train_mask_acc} | val_loss: {avg_val_loss} | val_mask_acc: {avg_val_mask_acc}')

  results[f'epoch_{epoch+1}'] = {
    'train_loss': avg_train_loss,
    'train_mask_acc': avg_train_mask_acc,
    'val_loss': avg_val_loss,
    'val_mask_acc': avg_val_mask_acc
  }
  ckpt_path = os.path.join(model_folder, f"epoch-{epoch+1}")
  model.save_pretrained(ckpt_path)

with open(config_path, 'w') as json_f:
  json.dump(Config, json_f)

with open(results_path, 'w') as json_f:
  json.dump(results, json_f)

end_time = datetime.now()
print(f"\n============= Total Training Time: {end_time - train_start_datetime} ============")
