# A Framework for Adapting Pre-Trained Language Models to Knowledge Graph Completion

This repository contains the official implementation for our paper:

**A Framework for Adapting Pre-Trained Language Models to Knowledge Graph Completion** \
Justin Lovelace and Carolyn Penstein Rosé \
In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022) 

## Dependencies

Our work was performed with Python 3.8. The dependencies can be installed from `requirements.txt`.

## Data Preparation

We conduct our work upon the FB15K-237, WN18RR, CN82K, and SNOMED-CT Core datasets.  Because the SNOMED-CT Core dataset was derived from the UMLS, we cannot directly release the dataset files. See [here](https://github.com/justinlovelace/robust-kg-completion) for full instructions for how to recreate the dataset. The other datasets can be processed with `python preprocessing/process_datasets.py`

We provide the BERT embeddings used from prior work (e.g. `data/CN82K/embeddings/bert-base-uncased_prior.pt`) used for our initial experiments as well as those extracted in our work (e.g. `data/CN82K/embeddings/bert-base-uncased-ft_mean.pt`).

## Candidate Embedding Processing
We provide example scripts for training a KGC model with our best unsupervised and supervised embedding processing techniques, the normalizing flow and the residual MLP in `scripts/tail_emb_processing/` for the CN82K dataset. The scripts can be applied to other datasets by updating the `--dataset` flag.

## Embedding Extraction Experiments
We provide example scripts for training a KGC model with our best supervised embedding extraction techniques, prompt tuning and linear probing in `scripts/head_ent_extraction/` for the CN82K dataset. The scripts can be applied to other datasets by updating the `--dataset` flag.

## Normalizing Flow Training
We provide an example script for training a normalizing flow to process embeddings at `flow/scripts/cn82k_flow.sh`. It should be run from the `flow/` directory.

## Acknowledgements

Our normalizing flow implementation was adapted from [chrischute](https://github.com/chrischute/glow)'s open-source implementation.