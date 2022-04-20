Config = {
  'case_sentence_csv_folder': r'',
  'csv_files': [
    'sentence_dataset_1000_cases.csv', 'sentence_dataset_2000_cases.csv', 'sentence_dataset_3000_cases.csv',
    'sentence_dataset_4000_cases.csv', 'sentence_dataset_5000_cases.csv', 'sentence_dataset_6000_cases.csv',
    'sentence_dataset_7000_cases.csv', 'sentence_dataset_8000_cases.csv', 'sentence_dataset_9000_cases.csv',
    'sentence_dataset_10000_cases.csv', 'sentence_dataset_11000_cases.csv', 'sentence_dataset_12000_cases.csv'
  ],
  'train_set_ratio': 0.8,
  'vocab_size': 30000,
  'seq_len': 128,
  'tokenizer_path': r'',
  'batch_size': 64,
  'embedding_dim': 768,
  'attention_heads': 8,
  'encoder_layers': 4,
  'intermediate_size': 1536,
  'type_vocab_size': 1,
  'learning_rate': 1e-4,
  'optimizer': 'AdamW',
  'model_path': r'',
  'start_epoch': 0,
  'num_epochs': 1,
  'last_ckpt_path': r''
}
