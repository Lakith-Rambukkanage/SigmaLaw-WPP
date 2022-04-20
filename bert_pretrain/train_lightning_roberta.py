import os
from datetime import datetime
import json
from transformers import RobertaConfig
from pytorch_lightning import Trainer

from pretrain_config import Config
from prepare_inputs import get_data_loader, get_mask_token_id
from lightning_roberta import LightningRoberta

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

dt_str = train_start_datetime.strftime("D%Y_%m_%d_T%H_%M_%S")
model_folder = os.path.join(Config['model_path'], dt_str)
ckpt_path = os.path.join(model_folder, 'checkpoints')
config_path = os.path.join(model_folder, 'train_config.json')
results_path = os.path.join(model_folder, 'results.json')

model = LightningRoberta(config, Config['last_ckpt_path'])

model.init_metrics()
model.set_learning_rate(Config['learning_rate'])
model.set_mask_token_id(mask_token_id)
model.set_ckpt_folder(ckpt_path)
model.set_start_epoch(Config['start_epoch'])

trainer = Trainer(
    devices='auto',
    # devices=4,
    accelerator='cpu',
    # strategy="ddp_spawn",
    max_epochs=Config['num_epochs'],
    enable_checkpointing=False
)

trainer.fit(model, train_dataloader, val_dataloader)

with open(config_path, 'w') as json_f:
  json.dump(Config, json_f)

results = {}
for key, value in model.train_results.items():
  results[key] = {
		'train_loss': value['loss'].item(),
		'train_mask_acc': value['mask_acc'].item(),
		'val_loss': model.val_results[key]['loss'].item(),
		'val_mask_acc': model.val_results[key]['mask_acc'].item()
  }

with open(results_path, 'w') as json_f:
  json.dump(results, json_f)

end_time = datetime.now()
print(f"\n============= Total Training Time: {end_time - train_start_datetime} ============")
