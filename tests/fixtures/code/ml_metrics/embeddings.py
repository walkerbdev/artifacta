"""Embedding visualization and analysis"""

import numpy as np
from sklearn.manifold import TSNE


def generate_embeddings(n_samples=500, n_dims=128):
    # Simulate embeddings from a trained model
    embeddings = np.random.randn(n_samples, n_dims)
    labels = np.random.randint(0, 5, n_samples)
    return embeddings, labels


def reduce_dimensionality(embeddings, method="tsne"):
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        return reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
