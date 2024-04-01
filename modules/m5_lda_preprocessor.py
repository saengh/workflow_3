from m1_main import *

import pandas as pd

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

def lda_preprocessor(text_fields_list, token_type, row, topic_num, topn_topics):

  # Creating a mini-corpus for the current row from preprocessed tokens
  mini_corpus = [row[f'{field}_{token_type}'] for field in text_fields_list]
  mini_corpus = [list(document) for document in mini_corpus]
  
  # Create dictionary and corpus for the mini-corpus
  dictionary = Dictionary(mini_corpus)
  corpus = [dictionary.doc2bow(document) for document in mini_corpus]
  
  # Running LDA on the mini-corpus
  lda_preprocessor_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_num, random_state=1,
                        update_every=0, passes=10, iterations=10, per_word_topics=True, minimum_probability=0.25)
  
  # Extract the main topic for the current document
  for bow in corpus:
    document_topics = lda_preprocessor_model.get_document_topics(bow)
    
    if document_topics:
      dominant_topic = max(document_topics, key=lambda x: x[1])[0]
      topic_prob = max(document_topics, key=lambda x: x[1])[1]
      topic_prob = f'{topic_prob: .2%}'
      topic_keywords = [word for word, prob in lda_preprocessor_model.show_topic(dominant_topic, topn=topn_topics)]
      topic_keywords_with_prob = [prob for word, prob in lda_preprocessor_model.show_topic(dominant_topic, topn=topn_topics)]
      # ", ".join([f'{prob:.2%} {word}' for word, prob in lda_preprocessor_model.show_topic(dominant_topic, topn=topn_topics)])
    else:
      dominant_topic = None
      topic_prob = "0%"
      topic_keywords = []
      topic_keywords_with_prob = "No dominant topic"

  return {
    'PNKC': row['PNKC'],
    f'Topic_Num_{token_type}': dominant_topic,
    f'Topic_Prob_{token_type}': str(topic_prob),
    f'Keywords_Prob_{token_type}': topic_keywords_with_prob,
    f'Keywords_{token_type}': topic_keywords
  }

def composite_topics(row):
  nested_topics = [row[topic_list] for topic_list in ['Keywords_word_tokens', 'Keywords_phrase_tokens']]
  flat_topics = [item for sublist in nested_topics for item in sublist]

  unique_topics = []

  for topic in flat_topics:
     if topic not in unique_topics:
        unique_topics.append(topic)

  return {
     'PNKC': row['PNKC'],
     'Composite_tokens': unique_topics
  }

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
  
df = pd.read_pickle(preprocessed_df_path)

unigrams_results_list = [lda_preprocessor(text_fields_list, token_type_list[0], row, 3, 50) for index, row in df.iterrows()]
bigrams_results_list = [lda_preprocessor(text_fields_list, token_type_list[1], row, 3, 50) for index, row in df.iterrows()]

unigrams_results_df = pd.DataFrame(unigrams_results_list)
bigrams_results_df = pd.DataFrame(bigrams_results_list)

# Validation step to check for empty lists in 'Topic_Keywords'
if any(unigrams_results_df[f'Keywords_{token_type_list[0]}'].apply(lambda x: x == [])):
    print(f"Warning: Empty lists found in 'Keywords_{token_type_list[0]}'")

if any(bigrams_results_df[f'Keywords_{token_type_list[1]}'].apply(lambda x: x == [])):
    print(f"Warning: Empty lists found in 'Keywords_{token_type_list[1]}'")

merged_df = pd.merge(df, unigrams_results_df, on='PNKC', how='left')
merged_df = pd.merge(merged_df, bigrams_results_df, on='PNKC', how='left')

composite_topics_list = [composite_topics(row) for index, row in merged_df.iterrows()]
composite_topics_df = pd.DataFrame(composite_topics_list)

merged_df = pd.merge(merged_df, composite_topics_df, on='PNKC', how='left')

columns_to_drop = [f"{field}_{token_type}" for field in fields for token_type in token_type_list]
merged_df = merged_df.drop(columns=columns_to_drop, errors='ignore')

merged_df.to_excel(workflow_folder + r'\excel\lda_preprocessed_df.xlsx', index=False)
merged_df.to_pickle(workflow_folder + r'\pickle\lda_preprocessed_df.pickle')