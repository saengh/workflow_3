from m1_main import *

import numpy as np
import pandas as pd

def generate_doc_topic_matrix(df, keyw_col, prob_col, token_type):

  # Get list of topics maintaining order
  topic_tokens = [list(topic_token) for topic_token in df[f'{keyw_col}_{token_type}']] # Create a list of list
  flat_topic_tokens = [item for sublist in topic_tokens for item in sublist] # Flaten the list of lists; used later

  # Get list of probabilities maintaining order
  prob_list = [list(prob) for prob in df[f'{prob_col}_{token_type}']] # Create a list of list

  # Create nested list of tuples of topic-prob
  combined_list = [list(zip(sublist_token, sublist_prob)) for sublist_token, sublist_prob in zip(topic_tokens, prob_list)]

  # Map nested tuple lists to document indices
  combined_dict = {}
  for sublist in combined_list:
    combined_dict[combined_list.index(sublist)] = sublist
    
  # Get unique list of token to serve as column header for the document_topix_matix
  unique_topic_tokens = []
  [unique_topic_tokens.append(topic_token) for topic_token in flat_topic_tokens if topic_token not in unique_topic_tokens]

  # Map unique tokens to their indices in the list
  column_to_index = {c: i for i, c in enumerate(unique_topic_tokens)}

  # Initialize the matrix with correct size
  document_token_matrix = np.zeros((len(df), len(unique_topic_tokens)))

  # Fill the matrix with probabilities
  for doc_index, values in combined_dict.items():
    for topic, prob in values:
      document_token_matrix[doc_index, column_to_index[topic]] = prob

  return document_token_matrix

df = pd.read_pickle(lda_preprocessed_df_path)

# Convert the matrix to a DataFrame for easier viewing and manipulation
df_unigrams_matrix = pd.DataFrame(generate_doc_topic_matrix(df, 'Keywords', 'Keywords_Prob', token_type_list[0]))
df_bigrams_matrix = pd.DataFrame(generate_doc_topic_matrix(df, 'Keywords', 'Keywords_Prob', token_type_list[1]))

# Generated matrix was too large for excel
# df_unigrams_matrix.to_excel(workflow_folder + r'\excel\document_topic_matrix_unigrams.xlsx')
# df_bigrams_matrix.to_excel(workflow_folder + r'\excel\document_topic_matrix_bigrams.xlsx')

df_unigrams_matrix.to_pickle(workflow_folder + r'\pickle\document_topic_matrix_unigrams.pickle')
df_bigrams_matrix.to_pickle(workflow_folder + r'\pickle\document_topic_matrix_bigrams.pickle')