## HypER: Literature-grounded Hypothesis Generation and Distillation with Provenance


This dataset supports the paper **"HypER: Literature-grounded Hypothesis Generation and Distillation with Provenance"** submitted to ARR 2025. It contains validated reasoning chains constructed from PubMed abstracts for hypothesis generation.

## Directory Structure

### `chain_data/batch*/`
- **`data/`**: selected reviews from RCT summarization data, and each review file includes associated PubMed paper details.
- **`invalid_chains_type1/`**: Chains where 10–50% of the intermediate nodes have been randomly replaced with irrelevant (score `0`) papers. These are "easy negatives."
- **`invalid_chains_type2/`**: Harder negative examples with coherent-looking subchains that contain logical breaks not easily detectable via surface similarity.
- **`result_chains/`**: Final valid chains where each chain includes paper titles, abstracts, publication years, relevance scores (`0=irrelevant`, `1=inspired`, `2=dependent`) with explanation.

### `source_data/`
Contains scripts and metadata for selecting **seed reviews** from approx. 4.5K RCT-based systematic reviews.

- `new_data_200_reviews.csv`: Metadata for 200 sampled reviews including title, abstract, and predicted subdiscipline.
- `document_predictions.json`: Model-generated subdiscipline predictions used for filtering and coverage balancing.
- `votek_sampling.py`: Implements the Vote-k sampling strategy to ensure diversity and avoid redundancy.
- `classify_reviews.py`: Classifies reviews into one of the four medical subdisciplines.

These source reviews serve as anchor points for building reasoning chains across four domains:
**Endocrinology**, **Cardiology**, **Rheumatology**, and **Gastroenterology/Hepatology**.

### `training_data/`
This folder contains multi-task fine-tuning data used to train the HypER model.

#### Contents
- `alpaca_multi_task3/alpaca_train.jsonl`  
- `alpaca_multi_task3/alpaca_valid.jsonl`  
- `alpaca_multi_task3/alpaca_test.jsonl`  

Each file contains reasoning chain samples labeled for:
- One-hop relevance classification  
- Multi-hop chain validation (agnostic & contextual)

#### Scripts
- `collect_paths.ipynb`: Extracts reasoning paths  
- `groundtruth_statistics.ipynb`: Computes dataset statistics  

