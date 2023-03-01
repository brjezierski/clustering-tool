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

from methods import run_clustering_alg, get_clustering_details, get_cluster_info, export_to_csv, read_cluster_labels, run_tsne, plot_2d, get_average_keyword_vector, embed, run_tsne_on_documents

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-kw", "--KEYWORD_FILE", help = "Input file of keyword vectors (required, format: pickled dict)")
    parser.add_argument("-doc", "--DOCUMENT_FILE", help = "Input file of documents (format: tsv with headers and documents in first column)")
    parser.add_argument("-k", "--K", help = "Number of clusters (required)")
    parser.add_argument("-cutoff", "--CUTOFF_THRESHOLD", help = "Cutoff threshold of cosine similarity to centroid, i.e. discard rows that do not fit well (range 0-1, default 0)")
    parser.add_argument("-weights", "--WEIGHTS_FILE", help = "File with weights (format: tsv with keywords in first column and weights in second)")
    parser.add_argument("-time", "--TIME", help = "Column name which contains time dimension to output how cluster sizes vary timewise")
    parser.add_argument('-col','--COLUMN', nargs='+', help='List of column names and integers for different vector encodings (1 - word embedding representation, 2 - one-hot), e.g., `-col keywords 1 categories 2` (default `keywords 1`)')
    parser.add_argument("-labels", "--LABELS_FILE", help = "File with cluster labels for the plot (format: tsv with cluster id in first column and label in second [label \"ignore\" to skip the cluster])")
    parser.add_argument('-plot', "--PLOT", action='store_true', help="Display the plot")
    parser.add_argument("-display", "--CLUSTERS_TO_DISPLAY", nargs='+', help = "List of cluster indices to display (List of cluster ids to display (works only with `-plot` flag, by default displays all clusters)")
    parser.add_argument('-p', "--PRINT", action='store_true', help="Print additional information")

    # Read arguments from command line
    args = parser.parse_args()
    to_print = args.PRINT
    cutoff_threshold = float(args.CUTOFF_THRESHOLD) if args.CUTOFF_THRESHOLD else 0.0

    clusters_to_display = []
    if args.CLUSTERS_TO_DISPLAY:
        for id in args.CLUSTERS_TO_DISPLAY:
            try:
                clusters_to_display.append(int(id))
            except ValueError:
                print("The clusters to display should be integer indices!")
                exit()

    algorithm = 'k-means' 

    labels = read_cluster_labels(args.LABELS_FILE) if args.LABELS_FILE else {}
    weights = pd.read_csv(args.WEIGHTS_FILE, sep='\t', names = ['kw', 'weight'], header=None) if args.WEIGHTS_FILE else None
    if weights is not None:
        weights = dict(zip(weights.kw, weights.weight))
    if args.K:
        k = int(args.K)
    else:
        print("Desired number of clusters missing!")
        exit()

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
    
        clustering, cluster_labels, cluster_scores, _ = run_clustering_alg(
            X=document_embeddings,
            k=k,
            algorithm=algorithm,
            print_metrics=to_print
        )

        clustering_details, reduced_model, reduced_cluster_labels = get_clustering_details(model, clustering, cluster_scores, algorithm=algorithm, cutoff_threshold=cutoff_threshold)
        most_rep_per_cluster = get_cluster_info(reduced_model, clustering, algorithm=algorithm, cutoff_threshold=cutoff_threshold, to_print=to_print, clustering_labels=reduced_cluster_labels)

        DIR = ("out")
        if not os.path.isdir(DIR):
            os.makedirs(DIR)
        clustered_df = pd.DataFrame(clustering_details, columns=["document", "cluster_id", "silhoutte_score", "avg_cos_sim_to_centroid", "cos_sim_to_centroid", "centroid_ratio"])
        weights_used = 'weighted.' if weights else ''
        cutoff_threshold_used = f'{cutoff_threshold}-threshold.' if cutoff_threshold > 0.0 else ''
        clustered_df.to_csv(f'out/{topic}.docs.{k}-clusters.{algorithm}.{columns_used}.{weights_used}{cutoff_threshold_used}csv', index=False)

        # Clustering plot
        if args.PLOT:
            reduced_vectors = run_tsne(reduced_model)
            plot_2d(k, reduced_vectors, reduced_cluster_labels, documents, most_rep_per_cluster, clusters_to_display=clusters_to_display, label_dict=labels) 
    
        # Time dimension
        if args.TIME:
            time_column = args.TIME
            docs_over_threshold = reduced_model.vocab.keys()
            document_df = document_df[document_df['document'].isin(docs_over_threshold)]

            time_analysis = [] 
            for m in unique(document_df[time_column]):
                tp_df = document_df.loc[document_df[time_column] == m]
                tp_model = KeyedVectors(vec_len)
                tp_embeddings = list(tp_df["encoding"])
                if to_print:
                    print(f"For quarter {m}: {len(tp_embeddings)} documents total")

                tp_clustering = clustering
                tp_clustering.labels_ = clustering.predict(tp_embeddings) 
                sizes = []
                for label in range(k):
                    size = np.count_nonzero(tp_clustering.labels_ == label)
                    sizes.append(int(size))
                sizes.insert(0, m)
                time_analysis.append(sizes)

            col_names = list(range(k))
            col_names.insert(0, time_column)
            df = pd.DataFrame(time_analysis, columns=col_names)
            df.to_csv(f'out/{topic}.time-analysis.docs.{k}-clusters.{algorithm}.{columns_used}.{weights_used}{cutoff_threshold_used}csv', index=False)
    
    # Clustering of keywords
    else:
        print(f"Clustering keywords from {args.KEYWORD_FILE}")
        clustering, cluster_labels, cluster_scores, _ = run_clustering_alg(
            X=vectorized_vocab,
            k=k,
            algorithm=algorithm,
            print_metrics=to_print
        )

        clustering_details, reduced_model, reduced_cluster_labels = get_clustering_details(model, clustering, cluster_scores, algorithm=algorithm, cutoff_threshold=cutoff_threshold)
        most_rep_per_cluster = get_cluster_info(model, clustering, algorithm=algorithm, to_print=to_print)
        DIR = ("out")
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

        export_to_csv(clustering_details, f"out/{topic}.kw.{k}-clusters.{algorithm}.csv", 'keyword')
        cluster_label_dict = read_cluster_labels(args.LABELS_FILE) if args.LABELS_FILE else {}

        # Clustering plot
        if args.PLOT:
            reduced_vectors = run_tsne(model)
            plot_2d(k, reduced_vectors, cluster_labels, vocab, most_rep_per_cluster, clusters_to_display=clusters_to_display, label_dict=cluster_label_dict)