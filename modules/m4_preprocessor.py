from m1_main import *

import pandas as pd

import re

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.corpora.dictionary import Dictionary
from gensim import corpora
from gensim.utils import simple_preprocess

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import download
download('punkt')
download('wordnet')
download('stopwords')

# keep alphabets only and convert to lower case 
def clean_text(text):
    text = re.sub("[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

# Tokenization, Lemmatization, removal of stopwords, and replacing synonyms
def word_tokenizer(text):    
    word_list = word_tokenize(text) # Tokenize
    return word_list

def word_lemmatizer(word_list):
    return [lemmatizer.lemmatize(word, pos='n') for word in word_list]

def stopword_remover(word_list):
    return [word for word in word_list if not (word in stop_words or len(word) <= 3)]

def synonym_replacer(word_list):
    return [synonyms[word] if word in synonyms else word for word in word_list]

# For executing sequence of word tokenization functions
def seq_word_tokens(row_text):    
    clean_row_text = clean_text(row_text)
    row_tokens = word_tokenizer(clean_row_text)
    reduced_row_tokens = stopword_remover(row_tokens)
    lemmatized_row_tokens = word_lemmatizer(reduced_row_tokens)
    swapped_row_tokens = synonym_replacer(lemmatized_row_tokens)
    final_rows_tokens = stopword_remover(swapped_row_tokens)

    return final_rows_tokens

# For generating phrase tokens only
def list_of_list_generator(df_column):
    df_column_list = [list(token_list) for token_list in df_column]
    return df_column_list

# For generating phrase tokens only
def phrase_tokenizer(list_of_word_tokens):
    # Train the Phrases model to automatically detect common phrases (bigrams)
    phrases = Phrases(list_of_word_tokens, min_count=1, threshold=1)
    bigram = Phraser(phrases)

    # Apply the trained model to transform patents into a list of phrases
    phrase_tokens = [bigram[phrase_tokens] for phrase_tokens in list_of_word_tokens]
    
    return phrase_tokens

# For generating sentence tokens only
def sent_tokenizer(row_text): 
    row_text = re.sub("[^a-zA-Z.\s]", "", row_text)
    row_text = row_text.lower()
    # row_text = re.sub("[a-zA-Z]{2}/", "", row_text)   
    sent_list = sent_tokenize(row_text) # Tokenize
    return sent_list

def clean_sent_tokens(sent_token_list):
    clean_sent_token_list = []
    for sent_token in sent_token_list:
        sent_token = re.sub("[^a-zA-Z\s]", "", sent_token)
        # Split the element into words if there's a space, and filter words longer than 3 characters
        if ' ' in sent_token:
            filtered_words = ' '.join(word for word in sent_token.split() if len(word) > 3)
            if len(filtered_words) > 3:
                clean_sent_token_list.append(filtered_words)
        elif len(sent_token) > 3:
            # Add the element if it's longer than 3 characters and doesn't contain a space
            clean_sent_token_list.append(sent_token)
    return clean_sent_token_list

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

# Load parsed dataset
df = pd.read_pickle(parsed_xml_cpc_path)

# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load in-built stopwords
stop_words = set(stopwords.words('english'))
lemmatized_stopwords = set(word_lemmatizer(stop_words))

# Load custom stopwords
custom_stopwords_df = pd.read_excel(custom_preprocessing_path, sheet_name="Stopwords")
custom_stopwords = set(custom_stopwords_df['Stopword'].str.strip())
lemmatized_custom_stopwords = set(word_lemmatizer(custom_stopwords))

#Load ignorewords
ignorewords_df = pd.read_excel(custom_preprocessing_path, sheet_name='Ignorewords')
ignorewords = set(ignorewords_df['Ignoreword'].str.strip())
lemmatized_ignorewords = set(word_lemmatizer(ignorewords))

# Update default stopwords with custom stopwords and ignorewords
stop_words.update(lemmatized_stopwords)
stop_words.update(custom_stopwords)
stop_words.update(lemmatized_custom_stopwords)
stop_words.update(ignorewords)
stop_words.update(lemmatized_ignorewords)

# Load synonyms
synonyms_df = pd.read_excel(custom_preprocessing_path, sheet_name='Synonyms')
synonyms = dict(zip(synonyms_df['Variant'].str.strip(), synonyms_df['Canonical'].str.strip()))

# Load words tokens into dataframe
for field in fields:
    df[f'{field}_word_tokens'] = df[field].apply(lambda row_text: seq_word_tokens(row_text) if isinstance(row_text, str) else [])
        # seq_word_tokens) # Assign the list of tokenized results to the new column in the dataframe

# Load phrase tokens into dataframe
for field in fields:
    df[f'{field}_phrase_tokens'] = phrase_tokenizer(list_of_list_generator(df[f'{field}_word_tokens']))

# Loading sentence tokens into DataFrame
for field in fields:
    all_sent_tokens = []
    for row_text in df[field]:
        if isinstance(row_text, str):
            processed_sent_tokens = clean_sent_tokens(sent_tokenizer(row_text))
            all_sent_tokens.append(processed_sent_tokens)
        else:
            all_sent_tokens.append([])
    df[f'{field}_sent_tokens'] = all_sent_tokens

df.to_excel(workflow_folder + r'\excel\preprocessed_df.xlsx', index=False)
df.to_pickle(workflow_folder + r'\pickle\preprocessed_df.pickle')

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

# TF-IDF - use for finding ignorewords only
from sklearn.feature_extraction.text import TfidfVectorizer

#  For TF-IDF which does not accept sets
tfidf_skip_list = list(stop_words)
df['CTB_word_tokens'] = df['CTB_word_tokens'].apply(lambda row: ' '.join(row))
df['CTB_CPC_word_tokens'] = df['CTB_CPC_word_tokens'].apply(lambda row: ' '.join(row))

tfidf_save_path = tfidf_matrix_path

tfidf_vectorizer = TfidfVectorizer(stop_words=tfidf_skip_list, ngram_range=(1,2), max_df=0.85, min_df=0.02, max_features=1000)

with pd.ExcelWriter(tfidf_save_path) as writer:
    for col in ['CTB_word_tokens', 'CTB_CPC_word_tokens']:
        tfidf_matrix = tfidf_vectorizer.fit_transform(df[col])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        tfidf_df.to_excel(writer, sheet_name=col)