# HypER: Literature-grounded Hypothesis Generation and Distillation with Provenance


## Overview

HypER is a multi-task fine-tuned SLM that learns to validate logical chains of scientific papers and generate hypotheses grounded in evidence-based reasoning. The model is trained on curated temporal reasoning chains from PubMed abstracts and demonstrates strong reasoning capabilities under noise.

---

## Dataset

The dataset includes:
- **3,523 reasoning chains** (valid, invalid-easy, invalid-hard)
- Each chain includes paper titles, abstracts, publication years, relevance scores (`0=irrelevant`, `1=inspired`, `2=dependent`) with explanation.
- Metadata with chain length, disruption level, and citation impact
- JSON format: `{ "chain_id": ..., "papers": [...], "labels": {...} }`

###

- collec_paths : to get summary of chains, split the data (summary and statistics of splits)
- 

#### Scripts
- `multi-task-data_prep.py`: Converts reasoning chains into multi-task format 
- `collect_paths.ipynb`: Summarizes chains and performs data splitting with statistics  
- `groundtruth_statistics.ipynb`: Computes dataset statistics  

