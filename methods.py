
import random
import torch
import os
import re
import string
import colorsys 

import nltk
import numpy as np
from numpy import unique
from numpy import where
import pandas as pd

from matplotlib import pyplot
from gensim.models import Word2Vec, KeyedVectors
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy import spatial
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestCentroid

import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.manifold import TSNE


def compute_cos_sim(vec1, vec2):
  return 1 - spatial.distance.cosine(vec1, vec2)

def get_centroid(clustering, embeddings=[], algorithm="k-means", clustering_labels=[]):
  if len(clustering_labels) == 0:
    clustering_labels = clustering.labels_
  if algorithm=="k-means":
    return clustering.cluster_centers_
  elif algorithm=="agglomerative":
    clf = NearestCentroid()
    clf.fit(embeddings, clustering_labels)
    return clf.centroids_
  else:
    print("Please choose an already implemented algorithm")
    return []

def get_avg_cos_sim_per_cluster(clustering, embeddings, algorithm):
  cluster_sizes = {}
  avg_cos_sim = {i:0 for i in unique(clustering.labels_)}
  centroids = get_centroid(clustering, embeddings, algorithm)
  for i in range(0, len(clustering.labels_)):
      cluster_id = clustering.labels_[i]
      word_score = compute_cos_sim(centroids[cluster_id], embeddings[i])#list(w2v_dict[word]))
      avg_cos_sim[cluster_id] += word_score
  total_avg = sum(avg_cos_sim.values())/len(clustering.labels_)
  for i in unique(clustering.labels_):
    avg_cos_sim[i] = avg_cos_sim[i]/np.count_nonzero(clustering.labels_ == i)
  return total_avg, avg_cos_sim

def get_offset(x, y):
  x_2 = 10 if x > 0 else -10
  y_2 = 25 if y > 0 else -25
  return (x_2, y_2)

def get_label(cluster, label_dict, kw):
  label = label_dict[cluster] if cluster in label_dict.keys() else kw
  return label if len(label) <= 80 else str(cluster)

def read_cluster_labels(filename):
  colnames = ['cluster_id', 'new_label']
  df = pd.read_csv(filename, sep='\t', names=colnames, header=None)
  return df.set_index('cluster_id').to_dict()['new_label']

def HSVToRGB(h, s, v): 
 (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
 return (int(255*r), int(255*g), int(255*b)) 
 
def getDistinctColors(n): 
 huePartition = 1.0 / (n + 1) 
 return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)) 

def get_color_dict(n):
  color_dict = {}
  i = 0
  for row in getDistinctColors(n):
    color_dict[i] = '#%02x%02x%02x' % row
    i += 1
  return color_dict

def plot_2d(k, reduced_embeddings, cluster_labels, vocab, most_representative, clusters_to_display = [], label_dict = {}):
    "Plots word embeddings reduced to two dimensions"
    x = []
    y = []
    for value in reduced_embeddings:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(10, 10)) 

    # plot the clusters
    for cluster in unique(cluster_labels):
      if cluster not in clusters_to_display and len(clusters_to_display) > 0:
        continue
      # get data points that fall in this cluster
      indices = where(cluster_labels == cluster)

      # make the plot
      if (len(indices) > 0):
        x_to_scatter = np.take(x, indices)
        y_to_scatter = np.take(y, indices)
        top_kw_ind = most_representative[cluster]
        label = get_label(cluster, label_dict, vocab[top_kw_ind])
        if label.lower() == "ignore":
          continue

        colors = get_color_dict(k)
        sc = pyplot.scatter(x_to_scatter, y_to_scatter, color=colors[cluster])
        plt.draw()
        # col = sc.get_facecolor()[0]
        plt.annotate(label,
                      xy=(x[top_kw_ind], y[top_kw_ind]),
                      xytext=get_offset(x[top_kw_ind], y[top_kw_ind]),
                      textcoords='offset pixels',
                      ha='right',
                      va='bottom',
                      fontsize=18,
                      arrowprops=dict(arrowstyle="->", color=colors[cluster]))
    plt.show()

def run_clustering_alg(
	  X, 
    k, 
    mb=500, 
    print_metrics=True,
    algorithm="k-means",
    seed=42,
):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_metrics: Print metrics values per cluster.
        algorithm: clustering algorithm to be used.

    Returns:
        Trained clustering model and labels based on X.
    """
    if algorithm == "k-means":
      clustering = MiniBatchKMeans(n_clusters=k, batch_size=mb, random_state=seed).fit(X)
    elif algorithm == "agglomerative":
      clustering = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    else:
      print("Please choose an already implemented algorithm")
      return
    total_avg_cos_sim, per_cluster_avg_cos_sim = get_avg_cos_sim_per_cluster(clustering, X, algorithm)

    clustering_scores = [silhouette_score(X, clustering.labels_), total_avg_cos_sim, calinski_harabasz_score(X, clustering.labels_), davies_bouldin_score(X, clustering.labels_)]
    if print_metrics:
        print(f"For n_clusters = {k}")
        print(f"Silhouette coefficient: {clustering_scores[0]:0.2f}")
        print(f"Average cosine similarity to the cluster center: {clustering_scores[1]:0.2f}")
        print(f"Calinski and Harabasz score: {clustering_scores[2]:0.2f}")
        print(f"Davies-Bouldin score: {clustering_scores[3]:0.2f}")

    cluster_scores = {}
    
    sample_silhouette_values = silhouette_samples(X, clustering.labels_)
    if print_metrics:
        print(f"Metric values:")
    metric_values = []
    for i in range(k):
        cluster_silhouette_values = sample_silhouette_values[clustering.labels_ == i]
        metric_values.append(
            (
                i,
                cluster_silhouette_values.shape[0],
                cluster_silhouette_values.mean(),
                per_cluster_avg_cos_sim[i],
                cluster_silhouette_values.min(),
                cluster_silhouette_values.max(),
            )
        )
        cluster_scores[i] = cluster_silhouette_values.mean()
    silhouette_values = sorted(
        metric_values, key=lambda tup: tup[2], reverse=True
    )
    if print_metrics:
        for s in metric_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg sil:{s[2]:.2f} | Avg cos sim:{s[3]:.2f}"
            )
    return clustering, clustering.labels_, cluster_scores, clustering_scores

def run_tsne(model, seed=42):
    "Reduces dimensions to 2 with t-SNE"
    labels = []
    tokens = []

    for word in model.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tokens = np.array(tokens)
    if torch.cuda.is_available():
      print("Running tSNE CUDA")
      new_values = TSNE_CUDA(n_components=2, perplexity=30, learning_rate=10, random_seed=seed).fit_transform(np.array(tokens))
    else:
      print("Running tSNE on CPU")
      tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1500, random_state=seed)
      new_values = tsne_model.fit_transform(tokens)

    return new_values

def run_tsne_on_documents(documents, seed=42):
    "Reduces dimensions to 2 with t-SNE"
  
    tokens = np.array(documents)
    if torch.cuda.is_available():
      print("Running tSNE CUDA")
      new_values = TSNE_CUDA(n_components=2, perplexity=30, learning_rate=10, random_seed=seed).fit_transform(np.array(tokens))
    else:
      print("Running tSNE on CPU")
      tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1500, random_state=seed)
      new_values = tsne_model.fit_transform(tokens)

    return new_values

def get_closest_point_to_centroid(centroid, points):
  points = np.array(points)
  centroid = np.array(centroid)
  distances = np.linalg.norm(points-centroid)
  min_index = np.argmin(distances)
  return points[min_index]

def get_clustering_details(model, clustering, cluster_scores, algorithm, cutoff_threshold=0.0):
  clustering_details = []
  vocab = [word for word in model.vocab]
  embeddings = [vec for vec in model.vectors]
  emb_dim = len(embeddings[0])
  reduced_vocab = []
  reduced_embeddings = []
  reduced_cluster_labels = []
  _, per_cluster_avg_cos_sim = get_avg_cos_sim_per_cluster(clustering, model.vectors, algorithm)

  for i in range(0, len(clustering.labels_)):
      tokens_per_cluster = ""
      word = vocab[i]
      vec = embeddings[i]
      cluster_id = clustering.labels_[i]
      centroids = get_centroid(clustering, model.vectors, algorithm)
      # get similarity score between the center of the cluster and the word's vector
      word_score = compute_cos_sim(centroids[cluster_id], model.vectors[i]) #list(w2v_dict[word]))
      centroid_ratio, cluster_2 = get_centroid_ratio(centroids, model.vectors[i])
      if max(word_score, 0) >= cutoff_threshold:
        reduced_vocab.append(word)
        reduced_embeddings.append(vec)
        reduced_cluster_labels.append(cluster_id)
        clustering_details.append([word, cluster_id, '%.2f' % cluster_scores[cluster_id], '%.2f' % per_cluster_avg_cos_sim[cluster_id], '%.2f' % word_score, '%.2f' % centroid_ratio])

  reduced_model = KeyedVectors(emb_dim)
  reduced_model.add(reduced_vocab, reduced_embeddings)
  return clustering_details, reduced_model, reduced_cluster_labels

def export_to_csv(clustering_details, filename, clustered_object):
  df = pd.DataFrame(clustering_details, columns=[clustered_object, "cluster_id", "silhoutte_score", "avg_cos_sim_to_centroid", "cos_sim_to_centroid", "centroid_ratio"])
  df.to_csv(filename, index=False)

def get_cluster_info(model, clustering, algorithm, topn_most_rep=5, to_print=True, cutoff_threshold=0.0, clustering_labels=[]):
  if to_print:
    print("Most representative terms per cluster (based on centroids):")
    print("Sidenote: these terms do no t necessarily belong to the given cluster, they are merely closest to the center of the cluster.")
  most_rep_per_cluster = {}
  vocab = [word for word in model.vocab]
  centroids = get_centroid(clustering, model.vectors, algorithm, clustering_labels=clustering_labels)
  for i in range(len(centroids)):
      tokens_per_cluster = ""
      most_representative = model.most_similar(positive=[centroids[i]], topn=topn_most_rep*5)
      count = 0
      for t in most_representative:
        if count >= topn_most_rep:
          break
        if t[1] > cutoff_threshold: # and clustering.labels_[vocab.index(t[0])] == i:
          tokens_per_cluster += f"\n\t{t[0]}[{t[1]}] "
          count += 1
      most_rep_per_cluster[i] = vocab.index(most_representative[0][0])
      if to_print:
        print(f"Cluster {i}: {tokens_per_cluster}")

  return most_rep_per_cluster

def print_top_labels_for_doc_clusters(topn, doc_df, clustered_doc_df, k, label="keywords"):
  print(f"Top {label} for each cluster")
  expanded_df = pd.merge(clustered_doc_df, doc_df, on='document', how='inner')
  new_df = expanded_df.explode(label)
  new_df = pd.crosstab(
        new_df[label], new_df['cluster_id']
    ).reset_index().rename_axis(columns=None)

  for i in range(0, k):
    print(f"Cluster {i}: {list(new_df.sort_values(by=i, ascending=False)[label].iloc[:topn])}")

def get_centroid_ratio(centroids, point):
  diff = centroids-np.tile(point,(centroids.shape[0], 1))
  dist = np.linalg.norm(diff, axis=1)
  idx = np.argpartition(dist, 2)
  return dist[idx[0]]/dist[idx[1]], idx[1]

def get_average_keyword_vector(keywords, vec_dict, weights=None):
  if not isinstance(keywords, list):
    return np.nan
  vectors = []
  weight_sum = 0.0
  for raw_kw in keywords:
    kw = raw_kw.replace(' ', '-')
    if weights is not None:
      if raw_kw in weights.keys() and kw in vec_dict.keys():
        vectors.append(weights[raw_kw]*vec_dict[kw])
        weight_sum += weights[raw_kw]
    else:
      if kw in vec_dict.keys():
        vectors.append(vec_dict[kw])
  if weights is not None:
    if weight_sum == 0.0:
        return np.nan
    out_vec = np.sum(vectors, axis=0)/weight_sum
  else:
    out_vec = np.average(vectors, axis=0)
  if not isinstance(out_vec, np.ndarray):
    return np.nan
  return out_vec

def embed(word_list, dictionary=None, weights=None):
  if dictionary is None:
    unique_words = {word: 0 for words in word_list for word in words}
    dictionary = {x: i for i, x in enumerate(unique_words)}
    
  idxss = [[dictionary[word] for word in words] for words in word_list]
  embedding = np.zeros((len(idxss), len(dictionary)), dtype=np.uint8)

  for i, idxs in enumerate(idxss):
    embedding[i, idxs] = 1

  return embedding, dictionary