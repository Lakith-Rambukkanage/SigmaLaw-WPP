Config = {
  'case_sentence_csv_folder': r'E:\Final Year Project\Datasets\criminal_sentences',
  'csv_file_list': [
    'sentence_dataset_1000_cases.csv', 'sentence_dataset_2000_cases.csv', 'sentence_dataset_3000_cases.csv',
    'sentence_dataset_4000_cases.csv', 'sentence_dataset_5000_cases.csv', 'sentence_dataset_6000_cases.csv',
    'sentence_dataset_7000_cases.csv', 'sentence_dataset_8000_cases.csv', 'sentence_dataset_9000_cases.csv',
    'sentence_dataset_10000_cases.csv',
  ],
  'batch_size': 4,
  'validation_split': 0.2,
  'vocab_size': 30000,
  'embed_dim': 300,
  'sequence_length': 128,
  'word_embeddings_type': 'GLOVE',
  'pretrained_word_embeddings_path': r'E:\Final Year Project\Datasets\word_embeddings\glove.6B',
  'encoder_units': 512,
  'decoder_units': 512,
  'recurrent_layer': 'GRU',
  'recurrent_layer_output_sequence': False,
  'loss_function': 'masked_loss',
  'accuracy_metric': 'sequence_regeneration_accuracy',
  'epochs': 3,
  'model_folder': r'E:\Final Year Project\case_sentence_embedding_models'
}