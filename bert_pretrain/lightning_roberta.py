import os
import torch
from transformers import RobertaForMaskedLM
from pytorch_lightning import LightningModule

from metrics_calc import masked_token_accuracy

class LightningRoberta(LightningModule):
  def __init__(self, config, pretrained_model_path=None):
    super().__init__()
    self.config = config
    if pretrained_model_path == None or pretrained_model_path == '':
      self.model = RobertaForMaskedLM(self.config)
    else:
      self.model = RobertaForMaskedLM.from_pretrained(pretrained_model_path, config=self.config)

  def set_learning_rate(self, lr):
    self.learning_rate = lr

  def set_mask_token_id(self, mask_id):
    self.mask_token_id = mask_id

  def forward(self, input_ids, attention_mask, labels=None):
    return self.model(input_ids, attention_mask=attention_mask, labels=labels)

  def training_step(self, batch, batch_idx):
    outputs = self(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
    masked_token_acc = masked_token_accuracy(outputs.logits, batch['input_ids'], batch['labels'], self.mask_token_id)
    self.log("train_loss", outputs.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log("train_mask_acc", masked_token_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return {'loss': outputs.loss, 'masked_token_acc': masked_token_acc}

  def validation_step(self, batch, batch_idx):
    outputs = self(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
    masked_token_acc = masked_token_accuracy(outputs.logits, batch['input_ids'], batch['labels'], self.mask_token_id)
    self.log_dict({'val_loss': outputs.loss, 'val_mask_acc': masked_token_acc}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return {'loss': outputs.loss, 'masked_token_acc': masked_token_acc}

  def configure_optimizers(self):
    return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

  def init_metrics(self):
    self.train_results = {}
    self.val_results = {}
    # self.train_metrics_log = {}
    # self.val_metrics_log = {}

  def set_ckpt_folder(self, ckpt_folder):
    self.ckpt_folder = ckpt_folder

  def set_start_epoch(self, epoch_num):
    self.start_epoch = epoch_num

  def training_epoch_end(self, training_step_outputs):
    # self.train_metrics_log[f'epoch_{self.start_epoch+self.current_epoch+1}'] = training_step_outputs
    total_loss = torch.tensor(0, dtype=training_step_outputs[0]['loss'].dtype, device=training_step_outputs[0]['loss'].device)
    total_acc = torch.tensor(0, dtype=training_step_outputs[0]['masked_token_acc'].dtype, device=training_step_outputs[0]['masked_token_acc'].device)
    for step_output in training_step_outputs:
      total_loss += step_output['loss']
      total_acc += step_output['masked_token_acc']
    self.train_results[f'epoch_{self.start_epoch+self.current_epoch+1}'] = {
        'loss': total_loss / len(training_step_outputs),
        'mask_acc': total_acc / len(training_step_outputs)
    }

  def validation_epoch_end(self, validation_step_outputs):
    # self.val_metrics_log[f'epoch_{self.start_epoch+self.current_epoch+1}'] = validation_step_outputs
    total_loss = torch.tensor(0, dtype=validation_step_outputs[0]['loss'].dtype, device=validation_step_outputs[0]['loss'].device)
    total_acc = torch.tensor(0, dtype=validation_step_outputs[0]['masked_token_acc'].dtype, device=validation_step_outputs[0]['masked_token_acc'].device)
    for step_output in validation_step_outputs:
      total_loss += step_output['loss']
      total_acc += step_output['masked_token_acc']
    self.val_results[f'epoch_{self.start_epoch+self.current_epoch+1}'] = {
        'loss': total_loss / len(validation_step_outputs),
        'mask_acc': total_acc / len(validation_step_outputs)
    }

  def on_train_epoch_end(self):
    completed_epoch = self.start_epoch + self.current_epoch + 1

    print_str = ">>>>"
    print_str += " train_loss: " + str(round(self.train_results[f'epoch_{completed_epoch}']['loss'].item(), 4))
    print_str += " | train_mask_acc: " + str(round(self.train_results[f'epoch_{completed_epoch}']['mask_acc'].item(), 4))
    print_str += " | val_loss: " + str(round(self.val_results[f'epoch_{completed_epoch}']['loss'].item(), 4))
    print_str += " | val_mask_acc: " + str(round(self.val_results[f'epoch_{completed_epoch}']['mask_acc'].item(), 4))

    ckpt_path = os.path.join(self.ckpt_folder, f"epoch-{completed_epoch}")
    self.model.save_pretrained(ckpt_path)

    print(print_str)
