## HypER: Literature-grounded Hypothesis Generation and Distillation with Provenance


This dataset supports the paper **"HypER: Literature-grounded Hypothesis Generation and Distillation with Provenance"** submitted to EMNLP 2025. It contains validated reasoning chains constructed from PubMed abstracts for hypothesis generation.

## Directory Structure

### `source_data/`
Contains scripts and metadata for selecting **seed reviews** from approx. 4.5K RCT-based systematic reviews.

#### Contents
- `new_data_200_reviews.csv`: Metadata for 200 sampled reviews including title, abstract, and predicted subdiscipline.
- `document_predictions.json`: Model-generated subdiscipline predictions used for filtering and coverage balancing.
- `votek_sampling.py`: Implements the Vote-k sampling strategy to ensure diversity and avoid redundancy.
- `classify_reviews.py`: Classifies reviews into one of the four medical subdisciplines.

These source reviews serve as anchor points for building reasoning chains across four domains:
**Endocrinology**, **Cardiology**, **Rheumatology**, and **Gastroenterology/Hepatology**.

### `chain_data/batch*/`
The folders `batch0` and `batch1` within `chain_data/` contains the core valid and invalid temporal reasoning chains to derive the training dataset for HypER.

#### Contents
- **`data/`**: selected seed reviews from RCT summarization data, and each review file includes associated PubMed paper details.
- **`intermediate_chains/`**: Intermediate reasoning chains with candidate nodes and their relevance scores. Nodes with score `0` (breakpoints) are used to generate invalid chains (`invalid_chains_type1/`, `invalid_chains_type2/`).
- **`invalid_chains_type1/`**: Chains where 10–50% of the intermediate nodes have been randomly replaced with irrelevant (score `0`) papers. These are "easy negatives."
- **`invalid_chains_type2/`**: Harder negative examples with coherent-looking subchains that contain logical breaks not easily detectable via surface similarity.
- **`result_chains/`**: Final valid chains where each chain includes paper titles, abstracts, publication years, relevance scores (`0=irrelevant`, `1=inspired`, `2=dependent`) with explanation.

### `training_data/`
This folder contains raw reasoning chain data and scripts used to generate multi-task fine-tuning datasets for the HypER model.

#### Contents
- `balanced_splits_w_hyp/`: Raw train/val/test splits of chains labeled with reasoning validity.
- `multi-task-data_prep.py`: Script to convert balanced chain data into multi-task format for training.
- `collect_paths.ipynb`: Extracts reasoning paths and creates chain-level summaries, also script for converting to alpaca formatted dataset.
- `groundtruth_statistics.ipynb`: Computes task and label distributions for analysis and construct the balanced dataset.

**Note that all raw data files are provided**, and the complete multi-task dataset can be reproduced using the included scripts.
