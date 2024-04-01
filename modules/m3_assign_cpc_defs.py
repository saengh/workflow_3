from m1_main import *

import pandas as pd

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

# Load parsed dataset
df = pd.read_pickle(parsed_xml_path)
cpc_df = pd.read_excel(cpc_defs_path)
merged_df = pd.merge(df, cpc_df, on='PNKC', how='left')

del merged_df['CPC']
merged_df.rename(columns={'CPC_DEFS': 'CPC'}, inplace=True)

# Create a combined fields (CTB, CPC) column
merged_df['CTB_CPC'] = merged_df[['CTB', 'CPC']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

merged_df.to_excel(workflow_folder + r'\excel\parsed_xml_cpc.xlsx', index=False)
merged_df.to_pickle(workflow_folder + r'\pickle\parsed_xml_cpc.pickle')