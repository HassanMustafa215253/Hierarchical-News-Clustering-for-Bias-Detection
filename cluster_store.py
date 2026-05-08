"""
cluster_store.py
────────────────
Saves all cluster artefacts produced by main() into a structured JSON file
(+ companion numpy .npz for arrays) so the search engine can reload them
without re-running the full pipeline.

Usage (add at the END of main(), after step 6):
    from cluster_store import save_cluster_data
    save_cluster_data(
        sentences       = sentences,
        embeddings      = embeddings,
        group_labels    = group_labels,
        group_topics    = group_topics,
        sub_labels      = sub_labels,
        combined_labels = combined_labels,
        combined_topics = combined_topics,
        out_dir         = CACHE_DIR,          # reuse same cache dir
    )
"""

from __future__ import annotations
import json
import os
import numpy as np


def save_cluster_data(
    sentences       : list[str],
    embeddings      : np.ndarray,
    group_labels    : np.ndarray,
    group_topics    : list[str],
    sub_labels      : np.ndarray,
    combined_labels : np.ndarray,
    combined_topics : dict[int, str],
    out_dir         : str = "/tmp/clustering_cache",
) -> dict[str, str]:
    """
    Persist all cluster artefacts.

    Saved files
    -----------
    cluster_data.npz  – numpy arrays (embeddings, group_labels, sub_labels, combined_labels)
    cluster_meta.json – everything that isn't a big array:
                          sentences, group_topics, combined_topics (keys as str)

    Returns a dict mapping artefact name → absolute path.
    """
    os.makedirs(out_dir, exist_ok=True)

    npz_path  = os.path.join(out_dir, "cluster_data.npz")
    json_path = os.path.join(out_dir, "cluster_meta.json")

    # ── 1. Arrays ──────────────────────────────────────────────────────────
    print("Saving cluster arrays …")
    np.savez_compressed(
        npz_path,
        embeddings      = embeddings.astype(np.float32),   # float32 halves disk use
        group_labels    = group_labels.astype(np.int32),
        sub_labels      = sub_labels.astype(np.int32),
        combined_labels = combined_labels.astype(np.int32),
    )
    print(f"  ✓ arrays  → {npz_path}  ({os.path.getsize(npz_path) / 1e6:.1f} MB)")

    # ── 2. Metadata ─────────────────────────────────────────────────────────
    print("Saving cluster metadata …")

    # Per-cluster centroid embeddings — stored as float32 lists
    # These are the search targets for cosine similarity.
    cluster_centroids: dict[str, list[float]] = {}
    for clabel in sorted(set(combined_labels)):
        if clabel == -1:
            continue
        mask = combined_labels == clabel
        centroid = embeddings[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)   # L2-normalise
        cluster_centroids[str(clabel)] = centroid.astype(np.float32).tolist()

    # Per-group centroid embeddings
    group_centroids: dict[str, list[float]] = {}
    for g, _ in enumerate(group_topics):
        mask = group_labels == g
        if not mask.any():
            continue
        centroid = embeddings[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        group_centroids[str(g)] = centroid.astype(np.float32).tolist()

    # Map combined_label → list of sentence indices (for retrieval)
    label_to_indices: dict[str, list[int]] = {}
    for i, lbl in enumerate(combined_labels.tolist()):
        key = str(lbl)
        label_to_indices.setdefault(key, []).append(i)

    meta = {
        "sentences"        : sentences,
        "group_topics"     : group_topics,
        # JSON keys must be strings; caller used int keys
        "combined_topics"  : {str(k): v for k, v in combined_topics.items()},
        "cluster_centroids": cluster_centroids,
        "group_centroids"  : group_centroids,
        "label_to_indices" : label_to_indices,
        "n_clusters"       : len(cluster_centroids),
        "n_groups"         : len(group_topics),
        "n_sentences"      : len(sentences),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"  ✓ metadata → {json_path}  ({os.path.getsize(json_path) / 1e6:.1f} MB)")
    print(f"\nCluster store ready:")
    print(f"  {meta['n_sentences']:,} sentences  |  {meta['n_groups']} groups  |  {meta['n_clusters']} clusters")

    return {"npz": npz_path, "json": json_path}
