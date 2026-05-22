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
    fuzzy_top_k     : int = 3,
    fuzzy_temperature: float = 0.08,
    fuzzy_min_similarity: float = 0.10,
) -> dict[str, str]:
    """
    Persist all cluster artefacts.

    embeddings must be the full 1024-D embedding space used for cosine similarity.
    Do not pass PCA-reduced purpose-space embeddings here.

    Saved files
    -----------
    cluster_data.npz  – numpy arrays (embeddings, group_labels, sub_labels, combined_labels)
    cluster_meta.json – everything that isn't a big array:
                          sentences, group_topics, combined_topics (keys as str),
                          fuzzy memberships, centroids, and label-to-indices map

    Returns a dict mapping artefact name → absolute path.
    """
    os.makedirs(out_dir, exist_ok=True)

    npz_path  = os.path.join(out_dir, "cluster_data.npz")
    json_path = os.path.join(out_dir, "cluster_meta.json")

    full_embeddings = embeddings

    # ── 1. Arrays ──────────────────────────────────────────────────────────
    print("Saving cluster arrays …")
    np.savez_compressed(
        npz_path,
        embeddings      = full_embeddings.astype(np.float32),   # float32 halves disk use
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
        centroid = full_embeddings[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)   # L2-normalise
        cluster_centroids[str(clabel)] = centroid.astype(np.float32).tolist()

    # Per-group centroid embeddings
    group_centroids: dict[str, list[float]] = {}
    for g, _ in enumerate(group_topics):
        mask = group_labels == g
        if not mask.any():
            continue
        centroid = full_embeddings[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        group_centroids[str(g)] = centroid.astype(np.float32).tolist()

    # Fuzzy memberships: doc -> top-k cluster ids + weights
    def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
        temp = max(temperature, 1e-6)
        z = x / temp
        z = z - np.max(z)
        exp = np.exp(z)
        return exp / (np.sum(exp) + 1e-9)

    centroid_items = sorted(cluster_centroids.items(), key=lambda kv: int(kv[0]))
    centroid_ids = np.array([int(k) for k, _ in centroid_items], dtype=np.int64)
    centroid_mat = np.array([v for _, v in centroid_items], dtype=np.float32)

    doc_memberships: list[list[list[float]]] = []
    label_to_indices: dict[str, list[int]] = {}

    if centroid_mat.size == 0:
        for i, lbl in enumerate(combined_labels.tolist()):
            doc_memberships.append([[int(lbl), 1.0]])
            label_to_indices.setdefault(str(lbl), []).append(i)
    else:
        chunk_size = 2000
        total = len(sentences)
        k = max(1, min(fuzzy_top_k, centroid_mat.shape[0]))

        for start in range(0, total, chunk_size):
            end = min(total, start + chunk_size)
            chunk = full_embeddings[start:end].astype(np.float32)
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            chunk = chunk / norms

            sims = chunk @ centroid_mat.T
            top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]

            for row in range(sims.shape[0]):
                idxs = top_idx[row]
                sims_i = sims[row, idxs]
                order = np.argsort(sims_i)[::-1]
                idxs = idxs[order]
                sims_i = sims_i[order]

                keep = sims_i >= fuzzy_min_similarity
                idxs = idxs[keep]
                sims_i = sims_i[keep]

                if idxs.size == 0:
                    best = int(np.argmax(sims[row]))
                    idxs = np.array([best], dtype=np.int64)
                    sims_i = np.array([sims[row, best]], dtype=np.float32)

                weights = _softmax(sims_i, fuzzy_temperature)
                memberships: list[list[float]] = []
                for j, w in zip(idxs.tolist(), weights.tolist()):
                    lbl = int(centroid_ids[j])
                    memberships.append([lbl, float(w)])
                    label_to_indices.setdefault(str(lbl), []).append(start + row)

                doc_memberships.append(memberships)

        # Deduplicate indices per label
        for k_lbl, v in list(label_to_indices.items()):
            label_to_indices[k_lbl] = sorted(set(v))

    meta = {
        "sentences"        : sentences,
        "group_topics"     : group_topics,
        # JSON keys must be strings; caller used int keys
        "combined_topics"  : {str(k): v for k, v in combined_topics.items()},
        "cluster_centroids": cluster_centroids,
        "group_centroids"  : group_centroids,
        "label_to_indices" : label_to_indices,
        "doc_memberships"  : doc_memberships,
        "membership_top_k" : int(fuzzy_top_k),
        "membership_temperature" : float(fuzzy_temperature),
        "membership_min_similarity" : float(fuzzy_min_similarity),
        "embedding_dim"    : int(full_embeddings.shape[1]) if hasattr(full_embeddings, "shape") else None,
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
