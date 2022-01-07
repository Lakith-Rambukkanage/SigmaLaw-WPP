import os

Config = {
  'case_sentence_csv_folder': r'E:\Final Year Project\Datasets\criminal_sentences',
  'csv_file_list': [
    'sentence_dataset_1000_cases.csv',
    # 'sentence_dataset_2000_cases.csv', 'sentence_dataset_3000_cases.csv',
    # 'sentence_dataset_4000_cases.csv', 'sentence_dataset_5000_cases.csv', 'sentence_dataset_6000_cases.csv',
    # 'sentence_dataset_7000_cases.csv', 'sentence_dataset_8000_cases.csv', 'sentence_dataset_9000_cases.csv',
    # 'sentence_dataset_10000_cases.csv',
  ],
  'batch_size': 4,
  'validation_split': 0.2,
  'vocab_size': 30000,
  'vocab_file': r'E:\Final Year Project\Datasets\bert_training\Tensorflow_TextVectorization_legal_vocab_30000\vocab.txt',
  'embed_dim': 300,
  'sequence_length': None,
  'word_embeddings_type': 'Glove',
  'pretrained_word_embeddings_path': r'E:\Final Year Project\Datasets\word_embeddings\glove.6B',
  'encoder_units': 512,
  'decoder_units': 512,
  'recurrent_layer': 'GRU',
  'recurrent_layer_output_sequence': False,
  'loss_function': 'mean_squared_error',
  'accuracy_metric': 'cosine_similarity',
  'use_nearest_token_embedding': False,
  'starting_epoch': 0,
  'num_epochs': 2,
  'learning_rate': 0.001,
  'checkpoints_per_epoch': 1,
  'logs_per_epoch': 20,
  'model_folder': r'E:\Final Year Project\case_sentence_embedding_models',
  'pre_trained_ckpt': None
}

def validate_config():
  dir_error = "Directory path not found!"
  assert os.path.isdir(Config['case_sentence_csv_folder']), \
    f"{dir_error} `case_sentence_csv_folder` should be the directory of case sentence files"
  assert os.path.isdir(Config['model_folder']), \
    f"{dir_error} `model_folder` should be the directory to save the checkpoints"
  assert os.path.isfile(Config['vocab_file']), \
    "File path not found! `vocab_file` should be the path to vocabulary file"

  assert len(Config['csv_file_list']) > 0, \
    "Empty file list! `csv_file_list` should contain file names from the `case_sentence_csv_folder`"

  assert isinstance(Config['validation_split'], float) and Config['validation_split'] < 0.5 and Config['validation_split'] > 0.1, \
    "Undesirable Value! `validation_split` should be a float value between 0.1 and 0.5"

  assert Config['recurrent_layer'] in ['GRU', 'LSTM'], \
    "Invalid option! `recurrent_layer` should be either `GRU` or `LSTM`"

  word_emb_types = ['Glove', 'Word2Vec', 'FastText', 'Glove_Legal', 'Word2Vec_Legal', 'FastText_Legal']
  assert Config['word_embeddings_type'] in word_emb_types, \
    f"Invalid option! `word_embeddings_type` should be either ${word_emb_types}"

  if Config['word_embeddings_type'] == 'Glove':
    assert os.path.isdir(Config['pretrained_word_embeddings_path']), \
      f"{dir_error} `pretrained_word_embeddings_path` should be the directory of Glove word embeddings"
  elif Config['word_embeddings_type'] in ['Word2Vec', 'FastText']:
    assert os.path.isfile(Config['pretrained_word_embeddings_path']), \
      f"Invalid file path! `pretrained_word_embeddings_path` should be the path to {Config['word_embeddings_type']} embeddings file"
