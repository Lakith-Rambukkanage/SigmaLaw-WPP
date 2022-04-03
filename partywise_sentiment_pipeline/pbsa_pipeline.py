import json
import gensim
from keras.models import load_model
from stanfordcorenlp import StanfordCoreNLP
from predict_parties_in_paragraph import predict_parties

GOOGLE_NEWS_VECTORS_PATH = r""
STANFORD_CORENLP = r""
GRU_512_MODEL = r""
GRU_512_MODEL_INPUT_LENGTH = 443

def get_stanford_annotater(stanford_core_nlp_path):
  nlp = StanfordCoreNLP(stanford_core_nlp_path, quiet=False)
  props = {'annotators': 'tokenize, ner, coref', 'pipelineLanguage': 'en'}
  return nlp, props

def get_word2vec_model(w2v_binary_model_path):
  return gensim.models.KeyedVectors.load_word2vec_format(w2v_binary_model_path, binary=True, unicode_errors='ignore')

def load_legal_party_prediction_model(model_path):
  return load_model(model_path)

def remove_non_ascii(text):
  return ''.join(i for i in text if ord(i)<128)

def get_token_words(annotated_txt):
  token_count = 0
  token_paragraph = []
  for sentence in annotated_txt['sentences']:
    token_words = []
    for token in sentence.get('tokens'):
      token_words.append(token['originalText'])
      token_count += 1
    token_paragraph.append(token_words)
  print("token count: ", token_count)
  return token_paragraph

def get_single_ref_locations(annotation, single_ref_petitioners, single_ref_defendants, locations_dict):
  for sent_ind in range(len(annotation['sentences'])):
    sentence_str = ""
    token_ind = 0
    for token in annotation['sentences'][sent_ind].get('tokens'):
      sentence_str += token['originalText'] + token['after']
      # print(sentence_str)

      for i in range(len(single_ref_petitioners)):
        if single_ref_petitioners[i] in sentence_str:
          if sent_ind not in locations_dict.keys():
            locations_dict[sent_ind] = {'petitioner': {}, 'defendant': {}}
          locations_dict[sent_ind]['petitioner'][token_ind] = single_ref_petitioners[i]
          del single_ref_petitioners[i]
          print(locations_dict)
          break

      for j in range(len(single_ref_defendants)):
        if single_ref_defendants[j] in sentence_str:
          if sent_ind not in locations_dict.keys():
            locations_dict[sent_ind] = {'petitioner': {}, 'defendant': {}}
          locations_dict[sent_ind]['defendant'][token_ind] = single_ref_defendants[j]
          del single_ref_defendants[j]
          print(locations_dict)
          break

      token_ind += 1

  return locations_dict

def get_petitioner_defendant_locations(annotation, petitioner_list, defendant_list):
  petitioner_coref_list = []
  defendant_coref_list = []
  entities_in_sentences = {}
  """
  {0: {'petitioner': {0: 'Dolores H. Cao', 7: 'her'}, 'defendant': {}},
   1: {'petitioner': {0: 'she', 14: 'her'}, 'defendant': {9: 'Puerto Rico Family Department'}},
  }
  """

  for coref_list in annotation['corefs'].values():
    entity = coref_list[0]['text']
    print("entity: ", entity)

    if entity in petitioner_list:
      party = 'petitioner'
      petitioner_coref_list.append(entity)
    elif entity in defendant_list:
      party = 'defendant'
      defendant_coref_list.append(entity)
    else:
      continue

    for item in coref_list:
      sentence_ind = item['sentNum'] - 1
      if sentence_ind not in entities_in_sentences.keys():
        entities_in_sentences[sentence_ind] = {'petitioner': {}, 'defendant': {}}
      entities_in_sentences[sentence_ind][party][item['startIndex'] - 1] = item['text']
      print(entities_in_sentences)

  print("petitioner_coref_list: ", petitioner_coref_list)
  print("defendant_coref_list: ", defendant_coref_list)
  remaining_petitioners = [pet for pet in petitioner_list if pet not in petitioner_coref_list]
  remaining_defendants = [defendant for defendant in defendant_list if defendant not in defendant_coref_list]

  print("======== Running: get_single_ref_locations() ....")

  final_entity_locations = get_single_ref_locations(annotation, remaining_petitioners, remaining_defendants, entities_in_sentences)
  return final_entity_locations

def add_occurance_count_for_repeated_elements(lst):
  """
  input = ['John', 'him', 'he', 'him', 'his', 'he']
  output = ['John', 'him_1', 'he_1', 'him_2', 'his', 'he_2']
  """
  for i in range(len(lst)):
    if lst.count(lst[i]) > 1:
      word = lst[i]
      same_word_count = 1
      for j in range(i, len(lst)):
        if lst[j] == word:
          lst[j] = f"{word}_{same_word_count}"
          same_word_count += 1
  return lst

def get_sentence_wise_entities_list(annotation, location_dict):
  ent_list = []
  for sent_ind in range(len(annotation['sentences'])):
    if sent_ind not in location_dict.keys():
      ent_list.append([[], []])
    else:
      sentence_entities = [[], []]
      for key in sorted(location_dict[sent_ind]['petitioner'].keys()):
        sentence_entities[0].append(location_dict[sent_ind]['petitioner'][key])
      for key in sorted(location_dict[sent_ind]['defendant'].keys()):
        sentence_entities[1].append(location_dict[sent_ind]['defendant'][key])

      sentence_entities[0] = add_occurance_count_for_repeated_elements(sentence_entities[0])
      sentence_entities[1] = add_occurance_count_for_repeated_elements(sentence_entities[1])
      ent_list.append(sentence_entities)
  return ent_list

def execute_pipeline(raw_text):
  ascii_text = remove_non_ascii(raw_text)

  nlp, props = get_stanford_annotater(STANFORD_CORENLP)
  annotation = json.loads(nlp.annotate(ascii_text, properties=props))

  w2v_model = get_word2vec_model(GOOGLE_NEWS_VECTORS_PATH)

  party_pred_model = load_legal_party_prediction_model(GRU_512_MODEL)

  petitioners_list, defendants_list = predict_parties(party_pred_model, w2v_model, annotation, ascii_text, GRU_512_MODEL_INPUT_LENGTH)
  
  entity_locations = get_petitioner_defendant_locations(annotation, petitioners_list, defendants_list)

  sentence_wise_entity_list = get_sentence_wise_entities_list(annotation, entity_locations)