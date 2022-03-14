# Equivariant-GNN-Word-Embeddings

CS 224N Final Project

## Installing

Relevant Packages can be installed using the requirements.txt file.

This project depends on the gvp-pytorch library as can be installed using https://github.com/drorlab/gvp-pytorch.

## Data

The ACE 2005 Dataset can be downloaded here: https://catalog.ldc.upenn.edu/LDC2006T06. A license is required.

The Drug Gen. Dataset can be downloaded
here: https://www.microsoft.com/en-us/research/project/project-hanover/downloads/.

The Billion Word Benchmark can be downloaded here: https://www.statmt.org/lm-benchmark/.

## Installing Benchmark repos
The file structure should look something like this:
```
├── data
│         ├── ACE2005-toolkit-main
│         │         ├── ace_2005 (put dataset here)
│         │         ├── cache_data
│         │         ├── filelist
│         │         ├── layered-bilstm-crf
│         │         │         ├── evaluation
│         │         │         ├── result
│         │         │         └── src
│         │         │             ├── dataset
│         │         │             └── model
│         │         ├── output (create empty, filled by processing)
│         │         └── udpipe
│         ├── billion
│         │         ├── heldout-monolingual.tokenized.shuffled
│         │         ├── splits (created by processing)
│         │         │         ├── dev
│         │         │         └── pretrain
│         │         └── training-monolingual.tokenized.shuffled
│         ├── drug
│         │         ├── data
│         │         ├── dist
│         │         └── machinereading
│         │                   ├── dist
│         │                   ├── evaluation
│         │                   └── models
│         ├── pubmed
│         │         ├── raw  (download xml here)
│         │         └── splits (made by process)
│         ├── pubmedword2vec (place here)
│         ├── walk-based-re
│         │         ├── configs
│         │         ├── data (processing makes this)
│         │         ├── data_processing
│         │         │         └── LSTM-ER
│         │         │                   ├── data
│         │         │                   │         └── ace2005
│         │         │                   │                   └── English (copy this from raw)
│         │         │                   ├── relation
│         │         │                   │         ├── cnn
│         │         │                   │         └── dict
│         │         │                   └── yaml
│         │         └── src
│         │             ├── analysis
│         │             ├── bin
│         │             └── nnet
│         └── wikipediaword2vec (copy here)
├── gvp-pytorch
│         ├── data
│         └── gvp
├── models
├── runs
│         ├── n_ary
│         ├── ner
│         ├── pretrain
│         └── re
├── scripts
└── utils
```

### Parsers
SciSpaCy model: https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz

### Word2Vec models
Can be downloaded, placed in correct directories, and loaded using Gensim binary functionality.

## Experiments
All experiments can be run using the scripts provided.

