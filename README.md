# Install libraries
```
pip install -r requirements.txt
```

# Command line format

| argv input                          | function     |
| ----------------------------------- | ------------ |
| `-kw KEYWORD_FILE`                  | Input file of keyword vectors (required, format: pickled dict) |
| `-doc DOCUMENT_FILE`                | Input file of documents (format: tsv with headers and documents in first column) |
| `-k K`                              | Number of clusters (required) |
| `-cutoff CUTOFF_THRESHOLD`          | Cutoff threshold of cosine similarity to centroid, i.e. discard rows that do not fit well (range 0-1, default 0) |
| `-weights WEIGHTS_FILE`             | File with weights (format: tsv with keywords in first column and weights in second) |
| `-time TIME`                        | Column name which contains time dimension to output how cluster sizes vary timewise |
| `-col COLUMN_NAME1 ENCODING1 ... ` | List of column names  and integers for different vector encodings (1 - word embedding representation, 2 - one-hot), e.g., `-col keywords 1 categories 2` (default `keywords 1`) |
| `-labels LABELS_FILE`               | File with cluster labels for the plot (format: tsv with cluster id in first column and label in second [label "ignore" to skip the cluster]) |
| `-plot`                             | Display the plot |
| `-p`                                | Print additional information |

# Examples of commands
```
python3 cluster.py -kw aerospace.kw.vec -k 40 -cutoff 0.5 -labels KeywordClusterNames.tsv -plot
python3 cluster.py -kw quantum.kw.vec -k 40 -cutoff 0.5 -plot
python3 cluster.py -kw quantum.kw.vec -doc quantum.tsv -k 40 -col keywords 1 categories 2 -weights bi_analysis.wkw.tsv -cutoff 0.8 -p  
python3 cluster.py -kw quantum.kw.vec -doc quantum.tsv -k 40 -col keywords 1
```

# Issues
  - clustering doesn't work for agglomerative clustering with threshold other than 0.0
  - the scores script needs to be completed
