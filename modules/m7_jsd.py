from m1_main import *

import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.spatial.distance import squareform, pdist
import numpy as np

def jsd(matrix):

  # Convert the DataFrame back to a numpy matrix for calculations, if it's not already
  matrix = matrix.to_numpy()
  smoothed_matrix = matrix + 1e-5
  smoothed_matrix /= smoothed_matrix.sum(axis=1, keepdims=True)

  # Calculating Jensen-Shannon Divergence between each pair of documents
  # Using pdist with a custom lambda function for JSD. Jensenshannon function expects 1D arrays for P and Q.
  jsd_matrix = squareform(pdist(smoothed_matrix, lambda u, v: jensenshannon(u, v)))

  # jsd_matrix now contains the pairwise distances. Let's convert it to a DataFrame for easier viewing and handling
  jsd_df = pd.DataFrame(jsd_matrix)

  return jsd_df

unigrams_matrix = pd.read_pickle(unigrams_matrix_path)
bigrams_matrix = pd.read_pickle(bigrams_matrix_path)

jsd_unigrams_df = jsd(unigrams_matrix)
jsd_bigrams_df = jsd(bigrams_matrix)

jsd_unigrams_df.to_excel(workflow_folder + r'\excel\jsd_unigrams.xlsx')
jsd_bigrams_df.to_excel(workflow_folder + r'\excel\jsd_bigrams.xlsx')

jsd_unigrams_df.to_pickle(workflow_folder + r'\pickle\jsd_unigrams.pickle')
jsd_bigrams_df.to_pickle(workflow_folder + r'\pickle\jsd_bigrams.pickle')
