import pandas as pd
from gensim.models import KeyedVectors

csv_path = ''   # golden standards csv path
keyed_vec_path = ''   # .kv file of word vectors

df = pd.read_csv(csv_path, header=None)
keyed_vec = KeyedVectors.load(keyed_vec_path, mmap='r')
word_limit = 6

match_ratios = []
for i in range(len(df)):
  # lower each word
  word = df.iloc[i, 0].lower()
  if word not in keyed_vec.index_to_key: continue
  similar_wrds = [x[0] for x in keyed_vec.most_similar(word, topn=10)]
  targets = df.iloc[i, 1:word_limit].tolist()
  total = 0
  match = 0
  for t in targets:
    t = t.lower()
    if t in keyed_vec.index_to_key:
      total += 1
      if t in similar_wrds: match += 1
  if total != 0: match_ratios.append(match/total)

print("Word Match Accuracy: ", sum(match_ratios) / len(match_ratios))