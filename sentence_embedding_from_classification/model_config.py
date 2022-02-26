dataset_config = {
  'case_sentence_csv_folder': r'E:\Final Year Project\Datasets\criminal_sentences',
  'csv_file_list': [
    'sentence_dataset_1000_cases.csv'
  ],
  'vocab_file': r'E:\Final Year Project\Datasets\vocab_files\Tensorflow_TextVectorization_legal_vocab_30000\vocab.txt',
  'general_words_file': r'E:\Final Year Project\Datasets\vocab_files\general_words.txt',
  'replacable_indices_file_path': r'E:\Final Year Project\Datasets\criminal_sentences\replacable_word_indices_1000.json',
  'sts_path': r'E:\Final Year Project\Datasets\sts_nli_annotated_data_for_legal_domain\legal_sts_merged.csv'
}

sentence_classification_config = {
  'seq_len': 64,
  'original_token_label': 0,
  'changed_token_label': 1,
  'changed_data_ratio': 0.5,
  'original_sentence_label': 0,
  'changed_sentence_label': 1,
  'replacing_ratio': 0.2
}

train_config = {
  'batch_size': 64,
  'train_set_ratio': 0.8,
  'val_set_ratio': 0.1,
  'num_classes': 2,
  'embed_dim': 300,
  'recurrent_layer': 'GRU',
  'rnn_units': 768,
  'starting_epoch': 0,
  'num_epochs': 10,
  'model_folder': r'E:\Final Year Project\case_sentence_embedding_models\noisy_sentence_discriminator_and_sts_task_model',
  'learning_rate': 0.001,
  'checkpoints_per_epoch': 1
}

sts_config = {
  'sts_batch_size': 16,
  'sts_train_ratio': 0.8,
  'sts_val_ratio': 0.1,
  'sts_loss_weight': 10
}

random_seeds = {
  'changed_sentence_selector': 507,
  'repl_word_selector': 23,
  'general_word_selector': 189,
  'shuffler': 15,
  'sts_shuffler': 56
}

def get_config_dict():
  config = {}
  config.update(dataset_config)
  config.update(sentence_classification_config)
  config.update(train_config)
  config.update(sts_config)
  config.update(random_seeds)
  return config