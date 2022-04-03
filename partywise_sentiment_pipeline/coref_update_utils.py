import copy

def build_complete_entity(ner, token_list):
  entity_words = []
  for token in token_list:
    if token['ner'] == ner:
      entity_words.append(token['originalText'])
    else:
      break
  return " ".join(entity_words)

def update_corefs(annotation):
  updated_corefs = copy.deepcopy(annotation['corefs'])
  for key in annotation['corefs'].keys():
    head_word_dict = annotation['corefs'][key][0]
    # print("head_word_dict: ", head_word_dict)
    sentence_tokens = annotation['sentences'][head_word_dict['sentNum']-1]
    head_word_tokens = sentence_tokens['tokens'][head_word_dict['startIndex']-1 : head_word_dict['endIndex']-1]
    # print("head_word_tokens: ", head_word_tokens)
    for i in range(len(head_word_tokens)):
      if head_word_tokens[i]['ner'] in ["PERSON", "ORGANIZATION", "LOCATION"]:
        head_word_ner = build_complete_entity(head_word_tokens[i]['ner'], head_word_tokens[i:])
        updated_corefs[key][0]['text'] = head_word_ner
        updated_corefs[key][0]['startIndex'] = head_word_dict['startIndex'] + i
        updated_corefs[key][0]['endIndex'] = head_word_dict['startIndex'] + i + len(head_word_ner.split())
        break
  
  return updated_corefs

def update_tokens_with_coref(annotation, updated_corefs):
  updated_tokens = copy.deepcopy(annotation['sentences'])
  for coref_list in updated_corefs.values():
    for i in range(len(coref_list)):
      ref = coref_list[i]
      for j in range(ref['startIndex']-1, ref['endIndex']-1):
        updated_tokens[ref['sentNum']-1]['tokens'][j]['coref'] = coref_list[0]['text']
        # print(updated_tokens[ref['sentNum']-1]['tokens'][j])
  return updated_tokens

def get_mask_value(mask_dict, word):
  for key in mask_dict.keys():
    if word in key:
      return mask_dict[key]

def get_entity_reference_list(token_list, ner_dict):
  entity_ref_list = []
  for token in token_list:
    if 'coref' in token.keys() and token['coref'] in ner_dict.keys():
      entity_ref_list.append(token['coref'])
    else:
      entity_ref_list.append(0)
  return entity_ref_list
