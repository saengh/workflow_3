from m1_main import *

import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim import corpora, models

unigrams_matrix = pd.read_pickle(unigrams_matrix_path)
bigrams_matrix = pd.read_pickle(bigrams_matrix_path)

# Hierarchical Clustering
n_clusters = 5
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
cluster_labels = clustering.fit_predict(bigrams_matrix)

# Visualization
pca = PCA(n_components=2)
doc_topic_2d = pca.fit_transform(bigrams_matrix)

plt.figure(figsize=(10, 8))
for cluster_num in range(n_clusters):
    plt.scatter(doc_topic_2d[cluster_labels == cluster_num, 0],
                doc_topic_2d[cluster_labels == cluster_num, 1],
                label=f'Cluster {cluster_num + 1}')

plt.title('Hierarchical Clustering of Patents')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.legend()
plt.show()

# Topic Summarization
corpus = [list(doc) for doc in unigrams_matrix]
dictionary = corpora.Dictionary(corpus)
lda_model = models.LdaModel(corpus, num_topics=n_clusters, id2word=dictionary)

top_words_per_topic = []
for topic in lda_model.print_topics():
    top_words_per_topic.append([word.split('*')[1].strip() for word in topic[1].split('+')])

for i, words in enumerate(top_words_per_topic):
    print(f"Cluster {i+1}: {', '.join(words)}")
