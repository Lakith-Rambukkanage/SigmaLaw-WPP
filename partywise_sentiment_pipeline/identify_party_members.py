import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd

from coref_update_utils import update_corefs, update_tokens_with_coref, get_entity_reference_list
from pbsa_pipeline import remove_non_ascii, add_occurance_count_for_repeated_elements, get_word2vec_model
from predict_parties_in_paragraph import makeTokenNERListsFromParagraph, getMaskValues, createVectorListFromToken
from keras.models import load_model
from stanfordcorenlp import StanfordCoreNLP

GOOGLE_NEWS_VECTORS_PATH = r""
STANFORD_CORENLP = r""
GRU_512_MODEL_UPDATED_COREF = r""
GRU_512_MODEL_INPUT_LENGTH = 443

output_folder = r""

case_sentence_csv_folder = r""
csv_file_list = [
  'sentence_dataset_1000_cases.csv', 'sentence_dataset_2000_cases.csv', 'sentence_dataset_3000_cases.csv',
  'sentence_dataset_4000_cases.csv', 'sentence_dataset_5000_cases.csv', 'sentence_dataset_6000_cases.csv',
  'sentence_dataset_7000_cases.csv', 'sentence_dataset_8000_cases.csv', 'sentence_dataset_9000_cases.csv',
  'sentence_dataset_10000_cases.csv',
]
file_index = 0

def get_stanford_annotater(stanford_core_nlp_path):
  nlp = StanfordCoreNLP(stanford_core_nlp_path, quiet=True, timeout=30000)
  props = {'annotators': 'tokenize, ner, coref', 'pipelineLanguage': 'en'}
  return nlp, props

def prepare_data(w2v_model, nlp, props, text):
  new_text = remove_non_ascii(text)

  annotation_result = json.loads(nlp.annotate(new_text, properties=props))
  updated_corefs = update_corefs(annotation_result)
  updated_tokens = update_tokens_with_coref(annotation_result, updated_corefs)

  ner_list, headwordsList, ner_dict, token_list = makeTokenNERListsFromParagraph(updated_tokens, updated_corefs)
  mask_dict = getMaskValues(ner_dict)

  vector_list = createVectorListFromToken(w2v_model, [token_list],[ner_list],[mask_dict])

  entity_ref_list = get_entity_reference_list(token_list, ner_dict)

  d = {x: 0 for x in ner_dict}

  # return new_text, d, headwordsList, vector_list
  return new_text, d, entity_ref_list, vector_list, annotation_result, token_list, updated_corefs

def pad_inputs(sentence_vectors_list, headwords_list, max_length=443):
  if len(headwords_list) > max_length:
    raise ValueError("tokens count exceeds the input sequence length of the model")

  padded_sentence_vectors_list = []
  for sentence_vectors in sentence_vectors_list:
    sentence_vectors.extend([np.zeros(301)] * (max_length - len(sentence_vectors)))
    padded_sentence_vectors_list.append(sentence_vectors)

  vectors_np = np.array(padded_sentence_vectors_list)
  # print("vectors_np shape: ", vectors_np.shape)

  headwords_list.extend([0] * (max_length - len(headwords_list)))

  return vectors_np, headwords_list

def get_sentence_start_token_indices(annotation):
  start_indices = [0]
  for i in range(len(annotation['sentences']) - 1):
    start_indices.append(start_indices[i] + len(annotation['sentences'][i]['tokens']))
  return start_indices

def get_sentence_wise_legal_entities(token_list, sentence_start_indices, petitioner_pred_list, defendant_pred_list):
  entity_list = {'petitioners': [], 'defendants': []}
  token_index = 0
  for i in range(len(sentence_start_indices)):
    if i == len(sentence_start_indices) - 1:
      end_index = len(token_list)
    else:
      end_index = sentence_start_indices[i+1]

    petitioner_token_words = []
    defendant_token_words = []
    while token_index < end_index:
      if petitioner_pred_list[token_index] >= 0.5 and defendant_pred_list[token_index] < 0.5:
        petitioner_entity = ""
        while petitioner_pred_list[token_index] >= 0.5 and defendant_pred_list[token_index] < 0.5:
          petitioner_entity += token_list[token_index]['originalText'] + token_list[token_index]['after']
          token_index += 1

        if token_index < end_index and token_list[token_index]['pos'] == "POS":
          petitioner_entity += token_list[token_index]['originalText']
          token_index += 1
        petitioner_token_words.append(petitioner_entity.strip())

      elif defendant_pred_list[token_index] >= 0.5 and petitioner_pred_list[token_index] < 0.5:
        defendant_entity = ""
        while defendant_pred_list[token_index] >= 0.5 and petitioner_pred_list[token_index] < 0.5:
          defendant_entity += token_list[token_index]['originalText'] + token_list[token_index]['after']
          token_index += 1

        if token_index < end_index and token_list[token_index]['pos'] == "POS":
          defendant_entity += token_list[token_index]['originalText']
          token_index += 1
        defendant_token_words.append(defendant_entity.strip())
      else:
        token_index += 1

    # print("petitioner_token_words: ", petitioner_token_words)
    # print("defendant_token_words: ", defendant_token_words)

    # entity_list['petitioners'] = add_occurance_count_for_repeated_elements(petitioner_token_words)
    # entity_list['defendants'] = add_occurance_count_for_repeated_elements(defendant_token_words)
    entity_list['petitioners'].extend(add_occurance_count_for_repeated_elements(petitioner_token_words))
    entity_list['defendants'].extend(add_occurance_count_for_repeated_elements(defendant_token_words))

  return entity_list

if __name__ == "__main__":
  df = pd.read_csv(os.path.join(case_sentence_csv_folder, csv_file_list[file_index]))
  nlp, props = get_stanford_annotater(STANFORD_CORENLP)
  w2v_model = get_word2vec_model(GOOGLE_NEWS_VECTORS_PATH)
  model = load_model(GRU_512_MODEL_UPDATED_COREF)
  party_members = {}
  for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # if index > 2: break
    # if not party_members.has_key(df['case_file']):
    #   party_members[df['case_file']] = []
    new_text, legal_entities, headwords_list, vector_list, annotation, token_list, updated_corefs = prepare_data(w2v_model, nlp, props, row['sentence'])
    vectors_np, headwords_padded = pad_inputs(vector_list, headwords_list)
    result = model.predict(vectors_np)
    petitioner_prob = result[0, :, 0]
    defendant_prob = result[0, :, 1]
    start_inds = get_sentence_start_token_indices(annotation)
    legal_entities = get_sentence_wise_legal_entities(token_list, start_inds,
                                                      petitioner_prob, defendant_prob)
    party_members[index] = legal_entities

  """ Write party members into json file """
  party_members_path = os.path.join(output_folder, f"party_members_case_{(file_index+1)*1000}.json")
  with open(party_members_path, 'w') as json_file:
    json.dump(party_members, json_file)
