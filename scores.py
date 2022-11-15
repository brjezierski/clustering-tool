import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
import matplotlib.pyplot as plt
from methods import run_clustering_alg
import argparse

def display_scores(plot, score, label, top_k):
  plot.plot(range(2, top_k), score[0:top_k-2])
  plot.set_title(label)
#   plt.xlabel("No. of clusters")
#   plt.ylabel("Scores")
#   plt.legend()
#   plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-kw", "--KEYWORD_FILE", help = "Input file of keyword vectors (required, format: pickled dict)")
    parser.add_argument("-doc", "--DOCUMENT_FILE", help = "Input file of documents (format: tsv with headers and documents in first column)") #TODO: test
    parser.add_argument("-weights", "--WEIGHTS_FILE", help = "File with weights (format: tsv with keywords in first column and weights in second)")
    parser.add_argument('-col','--COLUMN', nargs='+', help='List of column names  and integers for different vector encodings (1 - word embedding representation, 2 - one-hot), e.g., `-col keywords 1 categories 2` (default `keywords 1`)')
    # parser.add_argument("-alg", "--ALGORITHM", help = "Clustering algorithm (1 - k-means [default], 2 - agglomerative clustering)") #TODO: test
    parser.add_argument("-max", "--Max", help = "Choose top value of k to report scores (default 50)")

    # Read arguments from command line
    args = parser.parse_args()

    top_k = int(args.Max) if args.Max else 50
    algorithm = 'k-means' 
    weights = pd.read_csv(args.WEIGHTS_FILE, sep='\t', names = ['kw', 'weight'], header=None) if args.WEIGHTS_FILE else None
    if weights is not None:
        weights = dict(zip(weights.kw, weights.weight))

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
    

    if args.DOCUMENT_FILE:
        print(f"Clustering documents from {args.DOCUMENT_FILE}")
        topic = args.DOCUMENT_FILE.split('.')[0]

        document_df = pd.read_csv(args.DOCUMENT_FILE, sep='\t')
        document_df.drop_duplicates(subset=['document'], inplace=True)
        document_df["document"] = document_df["document"].str.strip()

        columns = {}
        if not args.COLUMN:
            columns["keywords"] = 1
        else:
            for i in range(len(args.COLUMN)):
                if i % 2 == 0:
                    column = args.COLUMN[i]
                else:
                    columns[column] = int(args.COLUMN[i])
        df_columns = []
        for column in columns:
            if columns[column] == 1:
                document_df[column] = document_df[column].str.split("|", expand = False)
                document_df[f"{column}_vec"] = [get_average_keyword_vector(keywords, w2v_dict, weights) for keywords in document_df[column]]
                document_df.dropna(inplace=True) 
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
        clustering, cluster_labels, cluster_scores, _ = run_clustering_alg(
            X=document_embeddings,
            k=k,
            algorithm=algorithm,
        )
    else:
        vectors = vectorized_vocab


    sil_scores = []
    cos_sim_scores = []
    cb_scores = []
    db_scores = []
    for k in range(2, top_k):
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

    fig, axs = plt.subplots(2, 2)

    display_scores(axs[0, 0], scores[0], 'Silhoutte', top_k)
    display_scores(axs[0, 1], scores[1], 'Average cosine similarity', top_k)
    display_scores(axs[1, 0], scores[2], 'Calinski and Harabasz', top_k)
    display_scores(axs[1, 1], scores[3], 'Davies-Bouldin', top_k)

    for ax in axs.flat:
        ax.set(xlabel="No. of clusters", ylabel="Scores")
    for ax in axs.flat:
        ax.label_outer()
    plt.legend()
    plt.show()