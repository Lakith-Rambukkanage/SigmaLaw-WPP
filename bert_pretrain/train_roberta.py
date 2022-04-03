import os
from datetime import datetime
import json
from tqdm import tqdm
import torch
from transformers import RobertaConfig, RobertaForMaskedLM, AdamW

from pretrain_config import Config
from bert_pretrain.prepare_inputs import get_data_loader

train_start_datetime = datetime.now()

train_dataloader, val_dataloader = get_data_loader()

config = RobertaConfig(
    vocab_size=Config['vocab_size'],  # we align this to the tokenizer vocab_size
    max_position_embeddings=Config['seq_len'],
    hidden_size=Config['embedding_dim'],
    num_attention_heads=Config['attention_heads'],
    num_hidden_layers=Config['encoder_layers'],
    intermediate_size=Config['intermediate_size'],
    type_vocab_size=Config['type_vocab_size']
)

model = RobertaForMaskedLM(config)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)

model.train() # activate training mode
optim = AdamW(model.parameters(), lr=Config['learning_rate'])

dt_str = train_start_datetime.strftime("D%Y_%m_%d_T%H_%M_%S")
model_folder = os.path.join(Config['model_path'], dt_str)
ckpt_path = os.path.join(model_folder, 'checkpoints')
config_path = os.path.join(model_folder, 'config.json')
results_path = os.path.join(model_folder, 'results.json')

start_epoch = Config['start_epoch']
end_epoch = start_epoch + Config['num_epochs']
results = {}

for epoch in range(start_epoch, end_epoch):
  # setup loop with TQDM and dataloader
  loop = tqdm(train_dataloader, leave=True)
  total_train_loss = 0.0
  for batch in loop:
    # initialize calculated gradients (from prev step)
    optim.zero_grad()
    # pull all tensor batches required for training
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    # process
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    # extract loss
    loss = outputs.loss
    # calculate loss for every parameter that needs grad update
    loss.backward()
    # update parameters
    optim.step()

    total_train_loss += loss.item()
    # print relevant info to progress bar
    loop.set_description(f'Epoch {epoch+1}')
    loop.set_postfix(loss=loss.item())

  avg_train_loss = total_train_loss / len(train_dataloader)

  total_val_loss = 0.0
  for batch in val_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    total_val_loss += outputs.loss.item()

  avg_val_loss = total_val_loss / len(val_dataloader)
  print(f'\nloss: {avg_train_loss} | val_loss: {avg_val_loss}')

  results[f'epoch_{epoch+1}'] = {
    'train_loss': avg_train_loss,
    'val_loss': avg_val_loss
  }
  ckpt_path = os.path.join(model_folder, f"epoch-{epoch+1}")
  model.save_pretrained(ckpt_path)

with open(config_path, 'w') as json_f:
  json.dump(Config, json_f)

with open(results_path, 'w') as json_f:
  json.dump(results, json_f)

end_time = datetime.now()
print(f"\n============= Total Training Time: {end_time - train_start_datetime} ============")
