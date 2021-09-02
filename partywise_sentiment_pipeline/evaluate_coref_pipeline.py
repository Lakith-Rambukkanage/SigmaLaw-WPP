def get_single_ref_pred_seq(annotation, single_ref_petitioners, single_ref_defendants, pet_coref_seq, def_coref_seq):
  for sent_ind in range(len(annotation['sentences'])):
    sentence_str = ""
    token_ind = 0
    for token in annotation['sentences'][sent_ind].get('tokens'):
      sentence_str += token['originalText'] + " "
      # print(sentence_str)
      
      for i in range(len(single_ref_petitioners)):
        if single_ref_petitioners[i] in sentence_str:
          # if sent_ind not in locations_dict.keys():
          #   locations_dict[sent_ind] = {'petitioner': {}, 'defendant': {}}
          # locations_dict[sent_ind]['petitioner'][token_ind] = single_ref_petitioners[i]
          num_words_in_entitiy = len(single_ref_petitioners[i].split(" "))
          pet_coref_seq[sent_ind][token_ind : token_ind + num_words_in_entitiy] = [1] * num_words_in_entitiy
          del single_ref_petitioners[i]
          # print(locations_dict)
          break

      for j in range(len(single_ref_defendants)):  
        if single_ref_defendants[j] in sentence_str:
          # if sent_ind not in locations_dict.keys():
          #   locations_dict[sent_ind] = {'petitioner': {}, 'defendant': {}}
          # locations_dict[sent_ind]['defendant'][token_ind] = single_ref_defendants[j]
          num_words_in_entitiy = len(single_ref_defendants[j].split(" "))
          def_coref_seq[sent_ind][token_ind : token_ind + num_words_in_entitiy] = [1] * num_words_in_entitiy
          del single_ref_defendants[j]
          # print(locations_dict)
          break

      token_ind += 1
  
  return pet_coref_seq, def_coref_seq

def get_coref_pred_seq(annotation, updated_corefs, petitioner_list, defendant_list, token_length=443):
  petitioner_coref_seq = []
  defendant_coref_seq = []

  petitioner_coref_list = []
  defendant_coref_list = []
  entities_in_sentences = {}
  
  for sentence in annotation['sentences']:
    petitioner_coref_seq.append([0] * len(sentence['tokens']))
    defendant_coref_seq.append([0] * len(sentence['tokens']))

  for coref_list in updated_corefs.values():
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
      # if sentence_ind not in entities_in_sentences.keys():
      #   entities_in_sentences[sentence_ind] = {'petitioner': {}, 'defendant': {}}
      # entities_in_sentences[sentence_ind][party][item['startIndex'] - 1] = item['text']
      if entity in petitioner_list:
        petitioner_coref_seq[sentence_ind][item['startIndex']-1 : item['endIndex']-1] = [1] * (item['endIndex'] - item['startIndex'])
      elif entity in defendant_list:
        defendant_coref_seq[sentence_ind][item['startIndex']-1 : item['endIndex']-1] = [1] * (item['endIndex'] - item['startIndex'])
      
      # print(entities_in_sentences)
  
  print("petitioner_coref_list: ", petitioner_coref_list)
  print("defendant_coref_list: ", defendant_coref_list)
  remaining_petitioners = [pet for pet in petitioner_list if pet not in petitioner_coref_list]
  remaining_defendants = [defendant for defendant in defendant_list if defendant not in defendant_coref_list]

  # print("======== Running: get_single_ref_pred_seq() ....")

  pet_coref_seq, def_coref_seq = get_single_ref_pred_seq(annotation, remaining_petitioners, remaining_defendants, petitioner_coref_seq, defendant_coref_seq)

  return pet_coref_seq, def_coref_seq

  # pet_coref_seq_1d = [val for sentence_seq in pet_coref_seq for val in sentence_seq]
  # def_coref_seq_1d = [val for sentence_seq in def_coref_seq for val in sentence_seq]

  # pet_coref_seq_1d.extend([0] * (token_length - len(pet_coref_seq_1d)))
  # def_coref_seq_1d.extend([0] * (token_length - len(def_coref_seq_1d)))

  # return pet_coref_seq_1d, def_coref_seq_1d
