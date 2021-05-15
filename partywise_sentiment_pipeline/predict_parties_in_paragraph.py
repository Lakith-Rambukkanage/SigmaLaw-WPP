import numpy as np

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

      if (current_ner_list[j] == "None"):
        try:
          vec = [np.append(w2v_model[current_token_list[j]], 0)]
          # print("current_ner_list[j] == 'None': ", vec)
        except KeyError:
          vec = [np.zeros(301)]
          # print("current_ner_list[j] == 'None': KeyError Occured!")

      else:
        try:
          # vec = [np.append(np.zeros(300), mask_dict[current_token_list[j]])]
          # vec = np.append(np.zeros(300), get_mask_value(mask_dict, current_token_list[j]))
          print("Running `get_entity_vector()`.... for word: ", current_token_list[j])
          vec, step = get_entity_vector(mask_dict, current_token_list, j)

          # print("current_ner_list[j] != 'None': ", vec)
        except KeyError:
          vec = [np.zeros(301)]
          print(f"current_ner_list[{j}] != 'None': KeyError Occured!")

      sentence_vector.extend(vec)
      j += step

    vector_list.append(sentence_vector)

  return vector_list

def makeTokenNERListsFromParagraph(text, annotation):
  token_list = []
  token_prefixes = []
  entity_prefixes = {}
  headwords_2d_list = []

  for sentence in annotation['sentences']:
    sentence_tokens = sentence.get('tokens')
    sentence_headwords = []

    for j in range(len(sentence_tokens)):
      token_list.append(sentence_tokens[j].get('word'))

      if (sentence_tokens[j].get("ner") == "PERSON" or sentence_tokens[j].get("ner") == "ORGANIZATION" or sentence_tokens[j].get("ner") == "LOCATION"):
        sentence_headwords.append(sentence_tokens[j].get('word'))
        full_name = build_complete_ner(sentence_tokens[j].get("word"), sentence_tokens[j].get("ner"), sentence_tokens[j+1 : len(sentence_tokens)])

        if (sentence_tokens[j].get("ner") == "PERSON"):
          prefix = "P"
        elif (sentence_tokens[j].get("ner") == "ORGANIZATION") :
          prefix = "O"
        else:
          prefix = "L"

        if (j==0 or (j!=0 and sentence_tokens[j-1].get("ner") != sentence_tokens[j].get("ner"))):
          entity_prefixes[full_name] = prefix
          words = full_name.split(" ")
          for l in words:
            token_prefixes.append(prefix)

      else:
        token_prefixes.append("None")
        sentence_headwords.append(0)

    headwords_2d_list.append(sentence_headwords)

  return token_list, token_prefixes, entity_prefixes, headwords_2d_list

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
          headwords_list[ref['position'][0] - 1][index] = coref_cluster[0]['text']
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