import numpy as np

from coref_update_utils import build_complete_entity

def build_complete_ner(word, ner, token_list):
  if (len(token_list)>0):
    if (token_list[0].get('ner') == ner):
      word = word + " " + build_complete_ner(token_list[0].get("word"), token_list[0].get('ner'), token_list[1:len(token_list)])
    #print("built" + word)
    return word
  else:
    return ""

def getMaskValues(ner_dict):
  masks_dict = {}
  keys = list(ner_dict.keys())
  for i in range(len(keys)):
    if ner_dict[keys[i]] == "P":
      value = 0.5*((1+i) / len(keys))
    elif ner_dict[keys[i]] == "O":
      value = 0.5 + 0.25*((1+i) / len(keys))
    elif ner_dict[keys[i]] == "L":
      value = 0.75 + 0.25*((1+i) / len(keys))
    masks_dict[keys[i]] = value
  return masks_dict

def get_entity_vector(mask_dict, token_list, current_index):
  for key in mask_dict.keys():
    entity_split = key.split()
    num_words = len(entity_split)
    if token_list[current_index] == entity_split[0]:
      if token_list[current_index: current_index + num_words] == entity_split:
        print("entity_split: ", entity_split)
        entity_vec = [np.append(np.zeros(300), mask_dict[key])] * num_words
        return entity_vec, num_words

  return [np.zeros(301)], 1

def createVectorListFromToken(w2v_model, token_list, ner_list, masks_dict_list):
  max_length = 0
  vector_list = []
  for i in range(0,len(token_list)):
    sentence_vector = []
    current_token_list = token_list[i]
    current_ner_list = ner_list[i]
    mask_dict = masks_dict_list[i]

    assert len(current_token_list) == len(current_ner_list)

    # for j in range(0, len(current_token_list)):
    j = 0
    while j < len(current_token_list):
      step = 1

      # if (current_ner_list[j] == "None"):
      if 'coref' not in current_token_list[j].keys():
        try:
          vec = [np.append(w2v_model[current_token_list[j]['originalText']], 0)]
          # print("current_ner_list[j] == 'None': ", vec)
        except KeyError:
          vec = [np.zeros(301)]
          # print("current_ner_list[j] == 'None': KeyError Occured!")

      else:
        try:
          # vec = [np.append(np.zeros(300), mask_dict[current_token_list[j]])]
          # vec = np.append(np.zeros(300), get_mask_value(mask_dict, current_token_list[j]))
          # print("Running `get_entity_vector()`.... for word: ", current_token_list[j])
          # vec, step = get_entity_vector(mask_dict, current_token_list, j)
          # print(f"word: {current_token_list[j]['originalText']} | mask value: {mask_dict[current_token_list[j]['coref']]}")
          if current_token_list[j]['ner'] in ["PERSON", "ORGANIZATION", "LOCATION"]:
            vec = [np.append(np.zeros(300), mask_dict[current_token_list[j]['coref']])]
          else:
            vec = [np.append(w2v_model[current_token_list[j]['originalText']], mask_dict[current_token_list[j]['coref']])]

          # print("current_ner_list[j] != 'None': ", vec)
        except KeyError:
          vec = [np.zeros(301)]
          # print(f"{current_token_list[j]['originalText']} : KeyError Occured!")

      sentence_vector.extend(vec)
      j += step

    vector_list.append(sentence_vector)

  return vector_list

def makeTokenNERListsFromParagraph(tokens_sentences, corefs):
  # result = json.loads(nlp.annotate(text, properties=props))
# def makeTokenNERListsFromParagraph(text, result):
  # sentences = result['sentences']
  # corefs = result['corefs']

  sentence_tokens_test = []
  token_list = []
  sentence_ners_test = []
  ner_test = {}

  for each in tokens_sentences:
    q = each.get('tokens')
    latest_entity = ""

    for j in range(len(q)):
      # sentence_tokens_test.append(q[j].get('word'))
      # token_list.append(q[j])

      if (q[j].get("ner") == "PERSON" or q[j].get("ner") == "ORGANIZATION" or q[j].get("ner") == "LOCATION"):
        # word = build_complete_ner(q[j].get("word"), q[j].get("ner"), q[j+1 : len(q)])
        word = build_complete_entity(q[j].get("ner"), q[j:])
        if (q[j].get("ner") == "PERSON"):
          prefix = "P"
        elif (q[j].get("ner") == "ORGANIZATION") :
          prefix = "O"
        else:
          prefix = "L"

        if (j==0):
          ner_test[word] = prefix
          words = word.split(" ")
          latest_entity = word
          for l in words:
            sentence_ners_test.append(prefix)

        if (j!=0 and q[j-1].get("ner") != q[j].get("ner")):
          ner_test[word] = prefix
          latest_entity = word
          words = word.split(" ")
          for l in words:
            sentence_ners_test.append(prefix)

        if 'coref' not in q[j].keys():
          q[j]['coref'] = latest_entity
          # print("single reference entity:", q[j])
        token_list.append(q[j])

      else:
        token_list.append(q[j])
        sentence_ners_test.append("None")

  # print("ner_test: ", ner_test)
  # print("sentence_ners_test: ", sentence_ners_test)

  headwords_list = []
  for sentence in tokens_sentences:
    headwords = []
    for each in sentence['tokens']:
      if(each.get("ner") == "PERSON" or each.get("ner") == "ORGANIZATION" or each.get("ner") == "LOCATION"):
        headwords.append(each.get('word'))
      else:
        headwords.append(0)

    headwords_list.append(headwords)

  # print("headwords_list: ", headwords_list)

  final_ner_dict = {}
  cluster_list = []

  for i in corefs.values():
    party_value = False
    for each in ner_test.keys():
      if each in i[0]['text']:
        party_value = True
        final_ner_dict[i[0]['text']] = ner_test[each]
    if party_value == True:
      cluster = []
      for j in i:
        # print(j)
        for index in range(j['startIndex']-1,j['endIndex']-1):
          # print(j['position'][0], index)
          headwords_list[j['position'][0]-1][index] = i[0]['text']
        cluster.append(j['text'])

      cluster_list.append(cluster)

  # print("final_ner_dict: ", final_ner_dict)
  # print("cluster_list: ", cluster_list)
  # print("headwords_list: ", headwords_list)

  final_headwords = []
  for i in headwords_list:
    for j in i:
      final_headwords.append(j)

  for ner in ner_test.keys():
    val = True
    for cluster_ner in final_ner_dict.keys():
      if ner in cluster_ner:
        val = False
    if (val):
      final_ner_dict[ner] = ner_test[ner]

  # return sentence_ners_test, final_headwords, final_ner_dict, sentence_tokens_test
  return sentence_ners_test, final_headwords, final_ner_dict, token_list

def getCorefTokens(entity_prefixes, headwords_2d_list, annotation):
  final_ner_dict = {}
  cluster_list = []

  for coref_cluster in annotation['corefs']:
    party_value = False
    for entity in entity_prefixes.keys():
      if entity in coref_cluster[0]['text']:
        party_value = True
        final_ner_dict[coref_cluster[0]['text']] = entity_prefixes[entity]
    if party_value == True:
      cluster = []
      for ref in coref_cluster:
        # print(ref)
        for index in range(ref['startIndex']-1, ref['endIndex']-1):
          # print(ref['position'][0], index)
          headwords_2d_list[ref['position'][0] - 1][index] = coref_cluster[0]['text']
        cluster.append(ref['text'])

      cluster_list.append(cluster)

  final_headwords = [headword for sentence in headwords_2d_list for headword in sentence]

  for ner in entity_prefixes.keys():
    val = True
    for cluster_ner in final_ner_dict.keys():
      if ner in cluster_ner:
        val = False
    if (val):
      final_ner_dict[ner] = entity_prefixes[ner]

  return final_ner_dict, final_headwords

def probabilityCal(legal_entities, headwords_list, plaintif_prob, defendant_prob):
  """
  Args:
    legal_entities: {'John H. Myers': 0, 'Insuarance Company': 0, ...}
    headwords_list: [coref for ner tokens; 0 for other tokens] <-- length = num. of tokens
    plaintif_prob: prediction probability for each token to be petitioner
    defendant_prob: prediction probability for each token to be defendant
  Output:
    list of petitioners, list of defendants
  """

  plaintifs = legal_entities.copy()
  defendants = legal_entities.copy()
  count = legal_entities.copy()
  plaintifs_list = []
  defendants_list = []

  for i in range(0,len(headwords_list)):
    try:
      plaintifs[headwords_list[i]] += plaintif_prob[i]
      defendants[headwords_list[i]] += defendant_prob[i]
      count[headwords_list[i]] += 1
    except KeyError:
      continue
    else:
      continue

  if (len(legal_entities) > 0):
    le_list=list(legal_entities)
    for j in range(len(legal_entities)):
      try:
        p=plaintifs[le_list[j]] / count[le_list[j]]
        d=defendants[le_list[j]] / count[le_list[j]]
      except ZeroDivisionError:
        p=0
        d=0
      print(le_list[j], " :", p, " , ", d)
      if(p>=0.5 and d<0.5):
        plaintifs_list.append(le_list[j])
      elif(d>=0.5 and p<0.5):
        defendants_list.append(le_list[j])
      elif(d<0.5 and p<0.5):
        continue
      elif(p>d):
        plaintifs_list.append(le_list[j])
      elif(d>p):
        defendants_list.append(le_list[j])

  return (plaintifs_list, defendants_list)

def pad_inputs(sentence_vectors_list, headwords_list, max_length=443):
  if len(headwords_list) > max_length:
    raise ValueError("tokens count exceeds the input sequence length of the model")

  padded_sentence_vectors_list = []
  for sentence_vectors in sentence_vectors_list:
    sentence_vectors.extend([np.zeros(301)] * (max_length - len(sentence_vectors)))
    padded_sentence_vectors_list.append(sentence_vectors)

  vectors_np = np.array(padded_sentence_vectors_list)
  print("vectors_np shape: ", vectors_np.shape)

  headwords_list.extend([0] * (max_length - len(headwords_list)))

  return vectors_np, headwords_list

def predict_parties(party_pred_model, w2v_model, annotation, ascii_text, model_input_length):
  token_list, token_prefixes, entity_prefixes, headwords_2d_list = makeTokenNERListsFromParagraph(ascii_text, annotation)

  entity_dict, headwords = getCorefTokens(entity_prefixes, headwords_2d_list, annotation)
  mask_dict = getMaskValues(entity_dict)

  vector_list = createVectorListFromToken(w2v_model, [token_list], [token_prefixes], [mask_dict])

  vectors_np, headwords_padded = pad_inputs(vector_list, headwords, model_input_length)

  pred_results = party_pred_model.predict(vectors_np)

  petitioner_probs = pred_results[:, :, 0].reshape(model_input_length).tolist()
  defendant_probs = pred_results[:, :, 1].reshape(model_input_length).tolist()

  petitioners_list, defendants_list = probabilityCal(
    {x: 0 for x in entity_dict},
    headwords_padded,
    petitioner_probs,
    defendant_probs
  )

  return petitioners_list, defendants_list