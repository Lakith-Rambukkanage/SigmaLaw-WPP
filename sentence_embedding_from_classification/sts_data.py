from random import Random
import pandas as pd
from model_config import dataset_config, sts_config
from prepare_data import preprocess

def get_sts_data():
  sts_df = pd.read_csv(dataset_config['sts_path'])
  sts_data = []
  for index, row in sts_df.iterrows():
    sts_data.append({
        'sentence1': preprocess(row['sentence1']),
        'sentence2': preprocess(row['sentence2']),
        'score': row['score']
    })
    sts_shuffler = Random(56)
    sts_train_ratio = sts_config['sts_train_ratio']
    sts_val_ratio = sts_config['sts_val_ratio']
    sts_shuffler.shuffle(sts_data)
    train_limit = int(sts_train_ratio * len(sts_data))
    val_limit = int((sts_train_ratio + sts_val_ratio) * len(sts_data))
    sts_train = sts_data[: train_limit]
    sts_val = sts_data[train_limit: val_limit]
    return sts_train, sts_val