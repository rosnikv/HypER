from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import seaborn as sns

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("document_predictions.json", "r") as f:
    data = json.load(f)

docs = [d["document"] for d in data]
labels = [d["predicted_specialty"] for d in data]
df = pd.DataFrame({"document": docs, "label": labels})
specialities = ["Endocrinology", "Cardiology", "Rheumatology", "Gastroenterology and Hepatology"]
df = df[df["label"].isin(specialities)].copy().reset_index(drop=True)
docs = df["document"].tolist()
doc_to_embeddings = model.encode(docs).tolist()
df["embeddings"] = doc_to_embeddings
#df = df.sample(frac=1, random_state=42).reset_index(drop=True)

def random_sampling(indices, num_samples):
    np.random.seed(42)
    return np.random.choice(indices, num_samples)

def fast_vote_k(embeddings, k, num_samples):
    n = len(embeddings)
    vote_stats = defaultdict(list)
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    for i in range(n):
        curr_embd = embeddings[i].reshape(1, -1)
        sims = cosine_similarity(embeddings, curr_embd).flatten()
        k_neighbors = np.argsort(sims)[-k:]
        for j in k_neighbors:
            if i != j:
                vote_stats[j].append(i)
    selected_indices =[]
    selected_times = defaultdict(int)
    #vote_stats = dict(sorted(vote_stats.items(), key = lambda x: len(x[1]), reverse=True))
    while len(selected_indices) < num_samples:
        scores = defaultdict(int)
        for idx, neighbors in vote_stats.items():
            if idx in selected_indices:
                scores[idx] = -100
                continue
            for n in neighbors:
                if n not in selected_indices:
                    scores[idx] += 10**(-selected_times[n])
        best_idx = max(scores, key=scores.get)
        selected_indices.append(best_idx)
        for n in vote_stats[best_idx]:
            selected_times[n] += 1
    return selected_indices

selected_indices = []
for sp in specialities:
    df_sp = df[df["label"] == sp]
    relative_indices = fast_vote_k(df_sp["embeddings"].tolist(), 20, 50)
    original_indices = df_sp.iloc[relative_indices].index.tolist()
    selected_indices += original_indices

new_df = df.iloc[selected_indices].copy().reset_index(drop=True)
tsne = TSNE(n_components=2, random_state=42)
embeddings = np.array(new_df["embeddings"].tolist())
embeddings_2d = tsne.fit_transform(embeddings)
new_df["x"] = embeddings_2d[:, 0]
new_df["y"] = embeddings_2d[:, 1]

"""
Random Sampling to see the difference
"""
random_indices = []
for sp in specialities:
    samples = random_sampling(df[df["label"] == sp].index.tolist(), 50)
    random_indices += list(samples)
random_df = df.iloc[random_indices].copy().reset_index(drop=True)
random_embeddings = np.array(random_df["embeddings"].tolist())
random_embeddings_2d = tsne.fit_transform(random_embeddings)
random_df["x"] = random_embeddings_2d[:, 0]
random_df["y"] = random_embeddings_2d[:, 1]

plt.figure(figsize=(8, 8))
sns.scatterplot(x="x", y="y", hue="label", data = new_df, legend="full")
plt.show()

new_df.to_csv("votek_sampling_results.csv", index=False)






