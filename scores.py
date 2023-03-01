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
from sklearn.manifold import TSNE

from methods import run_clustering_alg, get_clustering_details, get_cluster_info, export_to_csv, read_cluster_labels, run_tsne, plot_2d, get_average_keyword_vector, embed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def display_scores(plot, score, label, bottom_k, top_k):
  plot.plot(range(bottom_k, top_k), score[0:top_k-bottom_k])
  plot.set_title(label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-kw", "--KEYWORD_FILE", help = "Input file of keyword vectors (required, format: pickled dict)")
    parser.add_argument("-doc", "--DOCUMENT_FILE", help = "Input file of documents (format: tsv with headers and documents in first column)") #TODO: test
    parser.add_argument("-weights", "--WEIGHTS_FILE", help = "File with weights (format: tsv with keywords in first column and weights in second)")
    parser.add_argument('-col','--COLUMN', nargs='+', help='List of column names  and integers for different vector encodings (1 - word embedding representation, 2 - one-hot), e.g., `-col keywords 1 categories 2` (default `keywords 1`)')
    parser.add_argument("-min", "--Min", help = "Bottom value of k (default 2)")
    parser.add_argument("-max", "--Max", help = "Top value of k (default 50)")

    # Read arguments from command line
    args = parser.parse_args()

    top_k = int(args.Max) if args.Max else 50
    bottom_k = int(args.Min) if args.Min else 2
    if top_k < 2 or bottom_k < 2:
        print("Insert at least 2 clusters")
        exit()
    if top_k <= bottom_k:
        print("The max number of clusters has to be greater than the min")
        exit()

    algorithm = 'k-means' 
    weights = pd.read_csv(args.WEIGHTS_FILE, sep='\t', names = ['kw', 'weight'], header=None) if args.WEIGHTS_FILE else None
    if weights is not None:
        weights = dict(zip(weights.kw, weights.weight))

    # Prepare vocabolary with encodings
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
            print(f'Number of keywords: {len(vocab)}')
        else:
            print("%s vectors are empty!" % args.KEYWORD_FILE)
            exit()
    else:
        print("Input file missing!")
        exit()
    
    # Clustering of documents
    if args.DOCUMENT_FILE:
        print(f"Clustering documents from {args.DOCUMENT_FILE}")
        topic = args.DOCUMENT_FILE.split('.')[0]

        document_df = pd.read_csv(args.DOCUMENT_FILE, sep='\t')
        document_df.drop_duplicates(subset=['document'], inplace=True)
        document_df["document"] = document_df["document"].str.strip()

        columns = {}
        columns_used = ""
        if not args.COLUMN:
            columns["keywords"] = 1
            columns_used += "keywords-1"
        else:
            for i in range(len(args.COLUMN)):
                if i % 2 == 0:
                    column = args.COLUMN[i]
                    columns_used += column
                else:
                    columns[column] = int(args.COLUMN[i])
                    columns_used += f"-{columns[column]}_"
            columns_used = columns_used[:-1]

        df_columns = []
        for column in columns:
            if columns[column] == 1:
                document_df[column] = document_df[column].str.split("|", expand = False)
                document_df[f"{column}_vec"] = [get_average_keyword_vector(keywords, w2v_dict, weights) for keywords in document_df[column]]
                document_df.dropna(subset=['document', f'{column}_vec'], inplace=True)
                document_df[f'{column}_vec'] = document_df[f'{column}_vec'].apply(lambda x: list(x))
                df_columns.append(document_df[f'{column}_vec'])
            elif columns[column] == 2:
                document_df[column] = document_df[column].str.split("|", expand = False)
                document_df.dropna(inplace=True) 
                cat_embeddings, _ = embed(list(document_df[column]))
                document_df[f"{column}_vec"] = cat_embeddings.tolist()
                df_columns.append(document_df[f'{column}_vec'])
            else:
                print("Select one of the following encodings to use:\n\t1 - word embedding representation\n\t2 - one-hot")
                exit()

        for i in range(len(df_columns)):
            if i == 0:
                document_df["encoding"] = df_columns[i]
            else:
                document_df["encoding"] += df_columns[i]
        document_embeddings = list(document_df["encoding"])

        if len(document_embeddings) > 0:
            vec_len = len(document_embeddings[0])
            model = KeyedVectors(vec_len)
            documents = list(document_df["document"])
            model.add(documents, document_embeddings)
            print(f'Corpus size: {len(document_embeddings)}')
        else:
            print("%s vectors are empty!" % args.DOCUMENT_FILE)
            exit()
    
        vectors = document_embeddings
        weights_used = 'weighted.' if weights else ''
        title = f"{topic}.docs.{bottom_k}-{top_k}-clusters.{algorithm}.{columns_used}.{weights_used}"

    # Clustering of keywords
    else:
        vectors = vectorized_vocab
        title = f"{topic}.kw.{bottom_k}-{top_k}-clusters.{algorithm}."


    sil_scores = []
    cos_sim_scores = []
    cb_scores = []
    db_scores = []
    for k in range(bottom_k, top_k + 1):
        print(str(k) + " clusters\n")
        _, _, _, scores = run_clustering_alg(
            X=vectors,
            k=k,
            print_metrics=False,
        )
        sil_scores.append(scores[0])
        cos_sim_scores.append(scores[1])
        cb_scores.append(scores[2])
        db_scores.append(scores[3])
        
    scores = [sil_scores, cos_sim_scores, cb_scores, db_scores]

    fig, axs = plt.subplots(4, 1)

    display_scores(axs[0], scores[0], 'Silhoutte', bottom_k, top_k + 1)
    display_scores(axs[1], scores[1], 'Average cosine similarity', bottom_k, top_k + 1)
    display_scores(axs[2], scores[2], 'Calinski and Harabasz', bottom_k, top_k + 1)
    display_scores(axs[3], scores[3], 'Davies-Bouldin', bottom_k, top_k + 1)

    for ax in axs.flat:
        ax.set(xlabel="No. of clusters", ylabel="Scores")
    for ax in axs.flat:
        ax.label_outer()

    fig.set_size_inches(10, 14)
    plt.savefig(f'plots/{title}png')
    plt.legend()
    plt.show()