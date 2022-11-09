import random
import torch
import os
import re
import string
import colorsys
import argparse

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

from methods import run_clustering_alg, get_clustering_details, get_cluster_info, export_to_csv, read_cluster_labels, run_tsne, plot_2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-kw", "--KEYWORD_FILE", help = "Input file of keyword vectors") #TODO: test
    parser.add_argument("-doc", "--DOCUMENT_FILE", help = "Input file of documents") #TODO: test
    parser.add_argument("-k", "--K", help = "Number of clusters")
    parser.add_argument("-cutoff", "--CUTOFF_THRESHOLD", help = "Choose cutoff threshold (default 0.0)")
    parser.add_argument("-alg", "--ALGORITHM", help = "Select which clustering algorithm to use (1 - k-means [default], 2 - agglomerative clustering)") #TODO: test
    parser.add_argument("-labels", "--LABELS_FILE", help = "File with cluster labels") #TODO: test
    parser.add_argument('-plot', "--PLOT", action='store_true') #TODO: test
    parser.add_argument('-p', "--PRINT", action='store_true') #TODO: test

    # Read arguments from command line
    args = parser.parse_args()
    to_print = args.PRINT
    cutoff_threshold = float(args.CUTOFF_THRESHOLD) if args.CUTOFF_THRESHOLD else 0.0
    algorithm = args.ALGORITHM if args.ALGORITHM else 'k-means' 
    if algorithm == '2':
        algorithm = 'agglomerative'
    elif algorithm != 'k-means':
        print("Select one of the following algorithms:\n\t1 - k-means\n\t2 - agglomerative")
        exit()

    labels = read_cluster_labels(args.LABELS_FILE) if args.LABELS_FILE else {}
    weights = pd.read_csv(args.WEIGHTS_FILE, sep='\t', names = ['kw', 'weight'], header=None) if args.WEIGHTS_FILE else None
    if weights is not None:
        weights = dict(zip(weights.kw, weights.weight))
    if args.K:
        k = int(args.K)
    else:
        print("Desired number of clusters missing!")
        exit()
    
    if args.KEYWORD_FILE:
        print("Reading input file % s" % args.KEYWORD_FILE)
        topic = args.KEYWORD_FILE.split('.')[0]
        w2v_dict = pd.read_pickle(args.KEYWORD_FILE)
        vocab = list(w2v_dict.keys())
        vectorized_vocab = list(w2v_dict.values())

        if len(vectorized_vocab) > 0:
            vec_len = len(vectorized_vocab[0])
            model = KeyedVectors(vec_len)
            model.add(vocab, vectorized_vocab)
            print(f'{args.KEYWORD_FILE} vocabulary size: {len(vocab)}')
        else:
            print("%s vectors are empty!" % args.KEYWORD_FILE)
            exit()
    else:
        print("Input file missing!")
        exit()


    if args.K:
        k = int(args.K)
    else:
        print("Desired number of clusters missing!")
        exit()


    clustering, cluster_labels, cluster_scores, _ = run_clustering_alg(
        X=vectorized_vocab,
        k=k,
        algorithm=algorithm,
        print_metrics=to_print
    )

    clustering_details, indices_over_threshold, reduced_model, reduced_cluster_labels = get_clustering_details(model, clustering, cluster_scores, algorithm=algorithm, cutoff_threshold=cutoff_threshold)
    most_rep_per_cluster = get_cluster_info(model, clustering, algorithm=algorithm, cutoff_threshold=cutoff_threshold, to_print=to_print)
    DIR = ("out")
    if not os.path.isdir(DIR):
        os.makedirs(DIR)

    export_to_csv(clustering_details, f"out/{topic}.kw.{k}-clusters.{algorithm}.csv", 'keyword')
    cluster_label_dict = read_cluster_labels(args.Labels) if args.Labels else {}

    if args.Plot:
        reduced_vectors = run_tsne(model)
        plot_2d(k, reduced_vectors, cluster_labels, vocab, most_rep_per_cluster, clusters_to_display=[], indices_over_threshold=indices_over_threshold, label_dict=cluster_label_dict)
 
