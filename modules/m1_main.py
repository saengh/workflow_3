# Parent folder
parent_folder = r'C:\Users\SinghDis\3D Objects\my-files\ml-driven-landscapes'

# Workflow folder for saving
workflow_folder = r'C:\Users\SinghDis\3D Objects\my-files\ml-driven-landscapes\workflow_3'

# Document fields for xml_parser
input_xml_fields = ['ETI', 'EAB', 'ICLM', 'ECLM', 'CPC']

# Column names for dataframes
fields = ['TI', 'AB', 'ICLM', 'CLMS', 'CTB', 'CPC', 'CTB_CPC']
text_fields_list = ['TI', 'AB', 'ICLM', 'CLMS']
token_type_list = ['word_tokens', 'phrase_tokens', 'sent_tokens']

# File path for xml_parser
input_xml_path = workflow_folder + r'\xml\input_xml.xml'

# File path for shared resources
cpc_defs_path = parent_folder + r'\shared_resources\cpc_defs.xlsx'
custom_preprocessing_path = parent_folder + r'\shared_resources\custom_preprocessing.xlsx'

# # File paths for preprocessor
# parsed_xml_path = workflow_folder + r'\parquet\parsed_xml.parquet'
# parsed_xml_cpc_path = workflow_folder + r'\parquet\parsed_xml_cpc.parquet'
# preprocessed_df_path = workflow_folder + r'\parquet\preprocessed_df.parquet'
# lda_preprocessed_df_path = workflow_folder + r'\parquet\lda_preprocessed_df.parquet'
# unigrams_matrix_path = workflow_folder + r'\parquet\document_topic_matrix_unigrams.parquet'
# bigrams_matrix_path = workflow_folder + r'\parquet\document_topic_matrix_bigrams.parquet'

# File paths for preprocessor
parsed_xml_path = workflow_folder + r'\pickle\parsed_xml.pickle'
parsed_xml_cpc_path = workflow_folder + r'\pickle\parsed_xml_cpc.pickle'
preprocessed_df_path = workflow_folder + r'\pickle\preprocessed_df.pickle'
lda_preprocessed_df_path = workflow_folder + r'\pickle\lda_preprocessed_df.pickle'
unigrams_matrix_path = workflow_folder + r'\pickle\document_topic_matrix_unigrams.pickle'
bigrams_matrix_path = workflow_folder + r'\pickle\document_topic_matrix_bigrams.pickle'

tfidf_matrix_path = workflow_folder + r'\resources\tfidf_matrix.xlsx'