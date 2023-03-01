# Install libraries
```
conda env create -f environment.yml 
conda activate ct_env
```

# Clustering tool

## Description

This tool provides either a clustering of documents described by some attributes (keywords) or a clustering of keywords, using the K-means algorithm on the vector encoding of these keywords. 

## Command line format

| argv input                          | function     |
| ----------------------------------- | ------------ |
| `-kw KEYWORD_FILE`                  | Input file of vector encodings of keywords (required, format: pickled dict) |
| `-doc DOCUMENT_FILE`                | Input file of documents (format: tsv with headers, the documents are in the first column) |
| `-k K`                              | Number of clusters (required) |
| `-cutoff CUTOFF_THRESHOLD`          | Discard rows that do not fit well, throught a cutoff threshold based on the cosine similarity to the cluster centroid (range 0-1, default 0) |
| `-weights WEIGHTS_FILE`             | File that associates each keyword to a weight, in order to give a higher or lower importance to the keyword in the clustering (format: tsv with keywords in the first column and weights in the second) |
| `-time TIME`                        | Column name that contains the time dimension to output how cluster sizes vary timewise |
| `-col COLUMN_NAME1 ENCODING1 ... `  | List of columns to use for clustering and integers for different vector encodings (1 - word embedding representation, 2 - one-hot), e.g., `-col keywords 1 categories 2` (default `keywords 1`) |   
| `-plot`                             | Display the plot with the clusters, using the document most close to the centroid as a cluster label|
| `-labels LABELS_FILE`               | File that associates each cluster to a label. The plot will display these labels (format: tsv with cluster id in first column and label in second [label "ignore" to skip the cluster]) |     
| `-display ID_1 ID_2 ...`                          | List of cluster ids to display (works only with `-plot` flag, by default displays all clusters)    
| `-p`                                | Print on the terminal additional information such as clustering scores and the terms most closer to the centroid of each cluster|

## Clustering of keywords

All keywords present in the file passed after the argument `-kw` are clustered, according to their vector encodings.

## Examples of commands

To cluster keywords it is just necessary to not use the `-doc` argument

```
python3 cluster.py -kw aerospace.kw.vec -k 40 -cutoff 0.5 -labels aerospaceKeywordClusterNames.tsv -plot
python3 cluster.py -kw aerospace.kw.vec -k 40 -display 1 3 4 12 18 26 34 39 -plot
python3 cluster.py -kw quantum.kw.vec -k 40 -cutoff 0.5 -plot

```

## Clustering of documents

Documents can be anything that needs to be classified in different clusters, for example text documents or company names. These documents are clustered through their attributes, which can be for example relevant keywords and categories to cluster text documents, or company definitions to cluster companies. 

## Examples of commands

To cluster documents it is necessary to pass a document with the `-doc` argument and to specify the columns to use as attributes with the `-col` argument

```
python3 cluster.py -kw quantum.kw.vec -doc quantum.tsv -k 40 -col keywords 1 categories 2 -weights quantum_bi_analysis.wkw.tsv -cutoff 0.8 -p  
python3 cluster.py -kw quantum.kw.vec -doc quantum.tsv -k 40 -col keywords 1
python3 cluster.py -kw company.kw.vec -doc company_definitions.csv -k 30 -col definition 1 additional_definitions 1 -plot -p
python3 cluster.py -kw aerospace.kw.vec -k 40 -doc aerospace.tsv -col keywords 1 -cutoff 0.5 -time q -plot -p
```

## Output

The output is a csv file that contains each document, its cluster id, the silhoutte score of the cluster, the average cosine similarity to the centroid of the cluster, the cosine similarity to the centroid and the centroid ratio.
If the argument `-time` is used, an additional output file is generated, in which the first column indicates the times and the other columns indicate the size of each cluster, so that it is possible to visualize the variation of cluster sizes with time.

# Scores confrontation tool

## Description

This tool helps to find a suitable number of clusters for a set of documents or keywords. It produces clusterings from a minimum to a maximum number of clusters defined through command line, and it outputs a graph containing different scores (silhoutte score, average cosine similarity, Calinski and Harabasz index, Davies-Bouldin index) for every number of clusters. While these metrics can help choosing a fitting amount of clusters, it is still recommended to double check if the clustering is good both by looking at the plot generated by the `cluster.py` script with the `-plot` argument and by checking in the output file generated if entities in the same cluster are actually correlated.

## Command line format

| argv input                          | function     |
| ----------------------------------- | ------------ |
| `-kw KEYWORD_FILE`                  | Input file of vector encodings (required, format: pickled dict) |
| `-doc DOCUMENT_FILE`                | Input file of documents (format: tsv with headers and documents in first column) |
| `-weights WEIGHTS_FILE`             | File that associates each keyword to a weight, in order to give a higher or lower importance to the keyword in the clustering (format: tsv with keywords in the first column and weights in the second) |
| `-col COLUMN_NAME1 ENCODING1 ... `  | List of columns to use for clustering and integers for different vector encodings (1 - word embedding representation, 2 - one-hot), e.g., `-col keywords 1 categories 2` (default `keywords 1`) |  
| `-min`                               | Bottom value of k (default 2)
| `-max`                               | Top value of k (default 50)

## Examples of commands
```
python3 scores.py -kw company.kw.vec -doc company_definitions.csv -col definition 1 additional_definitions 1 -min 20 -max 60
python3 scores.py -kw company.kw.vec -max 20
```
## Output

The output is a graph that contains 4 different scores recorded for every number of clusters defined in the input. A way to choose the best numbers of clusters is to look at configurations that present peak values for multiple metrics used. The clustering is usually better when the silhoutte score, the average cosine similarity and the Calinski and Harabasz index have higher values, and when the Davies-Bouldin index has a lower value.