"""
cluster_search.py
─────────────────
Two-stage semantic search over saved cluster data, with bias detection.

  Stage 1 — Precomputed layer 1 retrieval
      Encode the user query with the same BGE model used during clustering.
      Rank precomputed group centroids by cosine similarity → return top-K groups.

  Stage 2 — Precomputed layer 2 retrieval
      Within each selected group, rank precomputed subcluster centroids.
      This narrows the search to the most relevant second-layer clusters.

  Stage 3 — Query-time refinement (up to 4 more layers)
      Recursion stops early when semantic homogeneity is detected
      (low intra-cluster variance means further splitting won't help).

  Stage 3 — Stance detection at leaf clusters
      Once a cluster is semantically tight (same event, different framing),
      split headlines by perspective using a stance/sentiment model.
      Returns BiasCluster objects pairing opposing viewpoints.

Usage
-----
    from cluster_search import ClusterSearch

    searcher = ClusterSearch(cache_dir="/tmp/clustering_cache")
    results  = searcher.search("covid vaccine side effects", top_k=5)
    searcher.display_results(results)
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
try:
    import faiss
except Exception:
    faiss = None

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Tunable constants
# ──────────────────────────────────────────────────────────────────────────────

# Cosine-similarity variance threshold for early exit from HDBSCAN recursion.
# If the average pairwise cosine distance inside a cluster is below this value,
# the cluster is already semantically tight — no further HDBSCAN split needed.
# Lower  → stricter (recurse more).  Higher → looser (stop earlier).
HOMOGENEITY_THRESHOLD = 0.08

# Minimum cluster size before we skip HDBSCAN and go straight to stance split.
MIN_STANCE_SIZE = 15

# Stance model — lightweight cross-encoder.
# Alternatives (heavier but more accurate):
#   "facebook/bart-large-mnli"
#   "cross-encoder/nli-deberta-v3-small"
STANCE_MODEL = "cross-encoder/nli-deberta-v3-small"

# Labels the NLI model is prompted with for stance polarity.
STANCE_LABELS = ["positive coverage", "negative coverage"]

# Max depth for HDBSCAN recursion after the two precomputed layers.
DEFAULT_MAX_DEPTH = 4


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LeafCluster:
    """
    A semantically coherent group of sentences returned by the search.

    depth=0  → top-level cluster (no further refinement done)
    depth>0  → emerged after `depth` rounds of HDBSCAN refinement
    has_stance_split=True → stance detection was run; see stance_sides
    nli_label/nli_score → per-document stance from NLI model
    """
    depth            : int
    topic            : str
    sentences        : list[str]
    similarity       : float
    parent_label     : int
    sub_label        : int          = -1
    has_stance_split : bool         = False
    stance_sides     : dict         = field(default_factory=dict)
    nli_label        : str | None   = None
    nli_score        : float        = 0.0

    def __repr__(self) -> str:
        snip = self.sentences[0][:80] + "…" if self.sentences else ""
        stance = " [BIAS SPLIT]" if self.has_stance_split else ""
        return (
            f"LeafCluster(depth={self.depth}, sim={self.similarity:.3f}, "
            f"n={len(self.sentences)}, topic='{self.topic[:50]}'{stance})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: TF-IDF topic inference
# ──────────────────────────────────────────────────────────────────────────────

def _infer_topic(sentences: list[str], top_terms: int = 5) -> str:
    from collections import Counter
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not sentences:
        return "miscellaneous"
    try:
        analyzer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2)
        ).build_analyzer()
        counts: Counter = Counter()
        for s in sentences:
            counts.update(analyzer(s))
        terms = [t for t, _ in counts.most_common(top_terms)]
        return ", ".join(terms) if terms else "miscellaneous"
    except Exception:
        return "miscellaneous"


# ──────────────────────────────────────────────────────────────────────────────
# Helper: semantic homogeneity check
# ──────────────────────────────────────────────────────────────────────────────

def _is_homogeneous(embeddings: np.ndarray, threshold: float = HOMOGENEITY_THRESHOLD) -> bool:
    """
    Returns True if the cluster is already semantically tight.

    Measured as mean cosine distance from each point to the centroid.
    If this is below `threshold`, further HDBSCAN splits are unlikely
    to produce meaningful topic separation — switch to stance detection instead.

    This is O(N*D) — fast even for large clusters.
    """
    if len(embeddings) < 4:
        return True

    # Unit-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    normed = embeddings / norms

    centroid = normed.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm < 1e-9:
        return False
    centroid /= centroid_norm

    # Cosine distances from centroid (1 - similarity)
    sims = normed @ centroid
    mean_dist = float(1.0 - sims.mean())

    return mean_dist < threshold


# ──────────────────────────────────────────────────────────────────────────────
# Helper: single-pass HDBSCAN (CPU)
# ──────────────────────────────────────────────────────────────────────────────

def _run_hdbscan(reduced: np.ndarray, group_size: int) -> np.ndarray:
    """
    Run HDBSCAN on already-reduced embeddings.
    Returns label array; -1 = noise.
    """
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        raise ImportError("pip install hdbscan")
    
    mcs      = max(3, min(group_size // 40, 40))
    min_samp = max(2, mcs // 3)

    labels = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=min_samp,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.3,
        metric="euclidean",
        core_dist_n_jobs=-1,
    ).fit_predict(reduced)

    return labels.astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: dimensionality reduction (mirrors main pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def _reduce(embeddings: np.ndarray, target_dims: int = 50) -> np.ndarray:
    """PCA-only reduction (UMAP removed). Returns float32 array."""
    from sklearn.decomposition import PCA

    N = embeddings.shape[0]
    if N < 4:
        return embeddings.astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    embeddings = embeddings / norms

    pca_dims = min(target_dims, N - 1, embeddings.shape[1])
    reduced = PCA(n_components=pca_dims).fit_transform(embeddings)
    return reduced.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: stance detection
# ──────────────────────────────────────────────────────────────────────────────

class _StanceDetector:
    """
    Lazy-loaded NLI cross-encoder for stance/framing detection.

    Uses zero-shot NLI: scores each sentence against
    "positive coverage" vs "negative coverage" as hypothesis labels.

    The model is shared across all ClusterSearch instances in a process.
    """
    _instance = None

    @classmethod
    def get(cls, model_name: str = STANCE_MODEL, device: str = "cpu") -> "_StanceDetector":
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name, device)
        return cls._instance

    def __init__(self, model_name: str, device: str) -> None:
        from transformers import pipeline as hf_pipeline
        print(f"  Loading stance model '{model_name}' on {device} …")
        self.model_name = model_name
        self._pipe = hf_pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )

    def score(self, sentences: list[str]) -> list[dict]:
        """
        Returns list of dicts:
          {"label": "positive coverage" | "negative coverage", "score": float}
        one per sentence.
        """
        if not sentences:
            return []

        results = self._pipe(
            sentences,
            candidate_labels=STANCE_LABELS,
            multi_label=False,
            batch_size=32,
        )

        # Normalise output to simple top-label dicts
        out = []
        for r in results:
            top_label = r["labels"][0]
            top_score = r["scores"][0]
            out.append({"label": top_label, "score": top_score})
        return out


def _split_by_stance(
    sentences : list[str],
    device    : str = "cpu",
    confidence: float = 0.60,
) -> dict:
    """
    Split a list of sentences into stance groups.

    Parameters
    ----------
    sentences  : headlines to classify
    device     : 'cuda' or 'cpu'
    confidence : minimum score to assign a stance label; below → 'neutral'

    Returns
    -------
    {
        "positive" : [sentences with positive framing],
        "negative" : [sentences with negative framing],
        "neutral"  : [sentences with no clear stance],
        "scores"   : [raw score dicts, one per sentence],
    }
    """
    detector = _StanceDetector.get(device=device)
    scored   = detector.score(sentences)

    groups: dict = {"positive": [], "negative": [], "neutral": [], "scores": scored}

    for sent, result in zip(sentences, scored):
        if result["score"] >= confidence:
            key = "positive" if result["label"] == "positive coverage" else "negative"
        else:
            key = "neutral"
        groups[key].append(sent)

    return groups


# ──────────────────────────────────────────────────────────────────────────────
# Main search engine
# ──────────────────────────────────────────────────────────────────────────────

class ClusterSearch:
    """
    Semantic cluster search with shallow HDBSCAN refinement and stance detection.

    Parameters
    ----------
    cache_dir      : directory where save_cluster_data() wrote its files.
    model_name     : BGE embedding model (must match the one used during indexing).
    device         : 'cuda' / 'cpu' / None (auto-detect).
    stance_model   : HuggingFace model for zero-shot stance classification.
    """

    def __init__(
        self,
        cache_dir    : str           = "/tmp/clustering_cache",
        model_name   : str           = "BAAI/bge-large-en-v1.5",
        device       : Optional[str] = None,
        stance_model : str           = STANCE_MODEL,
    ) -> None:
        self.cache_dir    = cache_dir
        self.model_name   = model_name
        self.stance_model = stance_model

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model         = None   # BGE encoder, lazy-loaded
        self._load_store()
        # FAISS index cache directory
        self._faiss_dir = os.path.join(self.cache_dir, "faiss_indexes")
        os.makedirs(self._faiss_dir, exist_ok=True)

    # ── Store loading ──────────────────────────────────────────────────────

    def _load_store(self) -> None:
        npz_path  = os.path.join(self.cache_dir, "cluster_data.npz")
        json_path = os.path.join(self.cache_dir, "cluster_meta.json")

        if not os.path.exists(npz_path) or not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Cluster store not found in '{self.cache_dir}'.\n"
                "Run save_cluster_data() first (see cluster_store.py)."
            )

        print(f"Loading cluster store from {self.cache_dir} …")
        npz = np.load(npz_path)
        self.embeddings      : np.ndarray     = npz["embeddings"]  # full 1024-D cosine space
        self.group_labels    : np.ndarray     = npz["group_labels"]
        self.sub_labels      : np.ndarray     = npz["sub_labels"]
        self.combined_labels : np.ndarray     = npz["combined_labels"]

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.sentences       : list[str]      = meta["sentences"]
        self.group_topics    : list[str]      = meta["group_topics"]
        self.combined_topics : dict[str, str] = meta["combined_topics"]
        self.label_to_indices: dict[str, list[int]] = meta["label_to_indices"]
        self.doc_memberships : list[list[list[float]]] | None = meta.get("doc_memberships")
        self.membership_top_k: int = int(meta.get("membership_top_k", 1))
        self.membership_min_similarity: float = float(meta.get("membership_min_similarity", 0.0))

        cluster_items = list(meta["cluster_centroids"].items())
        cluster_ids = [int(k) for k, _ in cluster_items]
        cluster_vecs = [v for _, v in cluster_items]

        self._cluster_centroid_ids : np.ndarray = np.array(cluster_ids, dtype=np.int64)
        self._cluster_centroid_mat : np.ndarray = np.array(cluster_vecs, dtype=np.float32)
        self._cluster_centroid_index: dict[int, int] = {
            int(lbl): i for i, lbl in enumerate(self._cluster_centroid_ids)
        }
        self._centroid_ids   = self._cluster_centroid_ids
        self._centroid_mat   = self._cluster_centroid_mat
        self._centroid_index = self._cluster_centroid_index

        group_items = list(meta.get("group_centroids", {}).items())
        if group_items:
            group_ids = [int(k) for k, _ in group_items]
            group_vecs = [v for _, v in group_items]
        else:
            group_ids = sorted({int(lbl // 10_000) for lbl in cluster_ids if lbl != -1})
            group_vecs = []
            for gid in group_ids:
                mask = self.group_labels == gid
                centroid = self.embeddings[mask].mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
                group_vecs.append(centroid.astype(np.float32).tolist())

        self._group_centroid_ids : np.ndarray = np.array(group_ids, dtype=np.int64)
        self._group_centroid_mat : np.ndarray = np.array(group_vecs, dtype=np.float32)
        self._group_centroid_index: dict[int, int] = {
            int(lbl): i for i, lbl in enumerate(self._group_centroid_ids)
        }

        self._group_to_cluster_ids: dict[int, list[int]] = {}
        for cid in self._cluster_centroid_ids.tolist():
            if cid == -1:
                continue
            gid = int(cid // 10_000)
            self._group_to_cluster_ids.setdefault(gid, []).append(int(cid))

        print(f"Loaded: {len(self.sentences):,} sentences, {meta['n_clusters']} clusters")

    # ── Embedding the query ────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """Encode query; returns unit-norm float32 vector shape (D,)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)

        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        vec = self._model.encode(
            [prefixed],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return vec.astype(np.float32)

    # ── FAISS helpers ─────────────────────────────────────────────────────

    def _faiss_index_path(self, label: int) -> str:
        return os.path.join(self._faiss_dir, f"{label}.index")

    def _faiss_map_path(self, label: int) -> str:
        return os.path.join(self._faiss_dir, f"{label}_docids.npy")

    def _build_faiss_index(self, label: int, nlist: int | None = None) -> None:
        """Build and persist a FAISS IVF index for the documents in `label`.

        This is lazy-built on demand. Uses IndexFlatIP quantizer + IndexIVFFlat
        with inner-product metric (embeddings are assumed L2-normalised).
        """
        if faiss is None:
            raise ImportError("faiss required. Install: pip install faiss-cpu")

        doc_ids = self.label_to_indices.get(str(label), [])
        if not doc_ids:
            raise ValueError(f"No documents found for label {label}")

        vecs = self.embeddings[np.array(doc_ids)].astype(np.float32)
        N, D = vecs.shape

        # Heuristic: choose nlist ~= sqrt(N) but at least 1
        if nlist is None:
            nlist = max(1, int(np.sqrt(max(1, N))))

        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)

        if not index.is_trained:
            index.train(vecs)
        index.add(vecs)

        # Persist index and docid map
        faiss.write_index(index, self._faiss_index_path(label))
        np.save(self._faiss_map_path(label), np.array(doc_ids, dtype=np.int64))

    def _load_faiss_index(self, label: int):
        """Load FAISS index for a label if present on disk; return (index, docids).
        Returns (None, None) if no index exists.
        """
        if faiss is None:
            return None, None
        p = self._faiss_index_path(label)
        m = self._faiss_map_path(label)
        if not os.path.exists(p) or not os.path.exists(m):
            return None, None
        idx = faiss.read_index(p)
        docids = np.load(m)
        return idx, docids

    def _ensure_faiss_index(self, label: int) -> tuple:
        """Ensure FAISS index exists for label; build if missing. Returns (index, docids)."""
        idx, docids = self._load_faiss_index(label)
        if idx is not None:
            return idx, docids
        # build (may be slow)
        self._build_faiss_index(label)
        return self._load_faiss_index(label)

    def _faiss_search_labels(
        self,
        query_vec: np.ndarray,
        labels: list[int],
        faiss_top_n: int = 200,
        nprobe: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Search FAISS indexes for a set of cluster labels and aggregate results.

        Returns (doc_ids, scores) arrays (both 1-D, aggregated across labels).
        """
        all_ids = []
        all_scores = []
        all_labels = []

        for lbl in labels:
            try:
                idx, docids = self._ensure_faiss_index(lbl)
            except Exception as e:
                print(f"  FAISS build/load failed for label {lbl} ({e}), skipping")
                continue
            if idx is None:
                continue
            # set nprobe for IVF
            try:
                idx.nprobe = nprobe
            except Exception:
                pass
            q = query_vec.reshape(1, -1).astype(np.float32)
            scores, local_ids = idx.search(q, faiss_top_n)
            scores = scores.ravel()
            local_ids = local_ids.ravel()
            # filter out -1 (missing)
            mask = local_ids >= 0
            local_ids = local_ids[mask]
            scores = scores[mask]
            # map to global doc ids
            global_ids = docids[local_ids]
            all_ids.append(global_ids)
            all_scores.append(scores)
            all_labels.append(np.full_like(global_ids, int(lbl), dtype=np.int64))

        if not all_ids:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.int64),
            )

        all_ids = np.concatenate(all_ids)
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        return all_ids, all_scores, all_labels

    # ── Retrieval over precomputed centroids ───────────────────────────────

    def _retrieve_clusters(
        self,
        query_vec: np.ndarray,
        top_k: int,
        min_similarity: float,
        centroid_ids: np.ndarray | None = None,
        centroid_mat: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        ids = self._centroid_ids if centroid_ids is None else centroid_ids
        mat = self._centroid_mat if centroid_mat is None else centroid_mat

        sims  = mat @ query_vec
        order = np.argsort(sims)[::-1]
        results = []
        for idx in order[:top_k]:
            sim = float(sims[idx])
            if sim < min_similarity:
                break
            results.append((int(ids[idx]), sim))
        return results

    def _retrieve_clusters_hybrid(
        self, query_vec: np.ndarray, top_k: int, min_similarity: float,
        top_sentences: int = 200,
    ) -> list[tuple[int, float]]:
        centroid_hits = self._retrieve_clusters(query_vec, top_k, min_similarity)

        sent_sims = self.embeddings @ query_vec
        top_sent_idx = np.argsort(sent_sims)[::-1][:top_sentences]

        from collections import Counter
        cluster_votes: Counter[int] = Counter()
        for si in top_sent_idx:
            if self.doc_memberships:
                memberships = self.doc_memberships[int(si)]
                for lbl, w in memberships:
                    cluster_votes[int(lbl)] += float(sent_sims[si]) * float(w)
            else:
                cl = int(self.combined_labels[si])
                if cl != -1:
                    cluster_votes[cl] += float(sent_sims[si])

        boosted: dict[int, float] = {}
        for lbl, sim in centroid_hits:
            vote_boost = cluster_votes.get(lbl, 0.0) / float(top_sentences)
            boosted[lbl] = sim + 0.15 * vote_boost

        for lbl, _score in cluster_votes.most_common(top_k * 2):
            if lbl in boosted:
                continue
            idx = self._centroid_index.get(lbl)
            if idx is None:
                continue
            centroid_sim = float(self._centroid_mat[idx] @ query_vec)
            if centroid_sim < min_similarity:
                continue
            boosted[lbl] = centroid_sim

        return sorted(boosted.items(), key=lambda x: -x[1])[:top_k]

    # ── Stage 2: shallow HDBSCAN with early exit ───────────────────────────

    def _recursive_refine(
        self,
        indices          : list[int],
        query_vec        : np.ndarray,
        parent_label     : int,
        parent_sim       : float,
        depth            : int,
        max_depth        : int,
        min_leaf_size    : int,
        run_stance       : bool,
        stance_confidence: float,
        top_k_per_level  : int = 3,   # ← new
    ) -> list[LeafCluster]:

        sents = [self.sentences[i] for i in indices]
        M     = len(indices)
        embs  = self.embeddings[np.array(indices)]

        def _return_leaf() -> list[LeafCluster]:
            return [self._make_leaf(
                indices, sents, depth, parent_label, -1,
                parent_sim, run_stance, stance_confidence,
            )]

        if M < 4:
            return _return_leaf()

        if depth >= max_depth:
            print(f"    [depth={depth}] depth limit — running stance split")
            return _return_leaf()

        if _is_homogeneous(embs, threshold=HOMOGENEITY_THRESHOLD):
            print(
                f"    [depth={depth}] cluster homogeneous (n={M}) "
                f"— skipping HDBSCAN, running stance split"
            )
            return _return_leaf()

        try:
            reduced = _reduce(embs, target_dims=min(50, M - 1))
            labels  = _run_hdbscan(reduced, group_size=M)
        except Exception as e:
            print(f"    [depth={depth}] HDBSCAN failed ({e}), returning as leaf")
            return _return_leaf()

        unique_subs = sorted(set(labels) - {-1})

        if len(unique_subs) <= 1:
            print(f"    [depth={depth}] HDBSCAN found no split (n={M}) — running stance split")
            return _return_leaf()

        # ── Score every sub-cluster by cosine similarity to query ─────────────
        scored_subs = []
        for sub_id in unique_subs:
            sub_idx  = [indices[i] for i, l in enumerate(labels) if l == sub_id]
            sub_centroid = self.embeddings[np.array(sub_idx)].mean(axis=0)
            norm = np.linalg.norm(sub_centroid)
            if norm > 1e-9:
                sub_centroid /= norm
            sub_sim = float(sub_centroid @ query_vec)
            scored_subs.append((sub_id, sub_idx, sub_sim))

        # ── Keep only top-K sub-clusters by similarity ─────────────────────────
        scored_subs.sort(key=lambda x: -x[2])
        kept  = scored_subs[:top_k_per_level]
        pruned = scored_subs[top_k_per_level:]

        if pruned:
            kept_str   = ", ".join(f"sub{s[0]}(sim={s[2]:.3f})" for s in kept)
            pruned_str = ", ".join(f"sub{s[0]}(sim={s[2]:.3f})" for s in pruned)
            print(
                f"    [depth={depth}] pruned {len(pruned)} sub-clusters "
                f"(keeping top {top_k_per_level} of {len(scored_subs)}): {kept_str}"
            )
            print(f"    [depth={depth}] pruned: {pruned_str}")

        leaves: list[LeafCluster] = []

        # Noise points — no recursion, no stance
        noise_idx = [indices[i] for i, l in enumerate(labels) if l == -1]
        if noise_idx:
            noise_sents = [self.sentences[i] for i in noise_idx]
            leaves.append(LeafCluster(
                depth=depth,
                topic=_infer_topic(noise_sents) + " [noise]",
                sentences=noise_sents,
                similarity=parent_sim,
                parent_label=parent_label,
                sub_label=-1,
            ))

        for sub_id, sub_idx, sub_sim in kept:
            if len(sub_idx) < min_leaf_size:
                sub_sents = [self.sentences[i] for i in sub_idx]
                leaves.append(LeafCluster(
                    depth=depth + 1,
                    topic=_infer_topic(sub_sents),
                    sentences=sub_sents,
                    similarity=sub_sim,
                    parent_label=parent_label,
                    sub_label=sub_id,
                ))
                continue

            children = self._recursive_refine(
                indices          = sub_idx,
                query_vec        = query_vec,
                parent_label     = parent_label,
                parent_sim       = sub_sim,
                depth            = depth + 1,
                max_depth        = max_depth,
                min_leaf_size    = min_leaf_size,
                run_stance       = run_stance,
                stance_confidence= stance_confidence,
                top_k_per_level  = top_k_per_level,   # ← pass down
            )
            leaves.extend(children)

        return leaves

    # ── Leaf builder (applies stance detection when appropriate) ───────────

    def _make_leaf(
        self,
        indices          : list[int],
        sents            : list[str],
        depth            : int,
        parent_label     : int,
        sub_label        : int,
        similarity       : float,
        run_stance       : bool,
        stance_confidence: float,
    ) -> LeafCluster:
        """
        Build a LeafCluster, optionally applying stance detection.

        Stance detection is applied when:
          - run_stance=True (user opted in)
          - cluster has >= MIN_STANCE_SIZE sentences (enough to split meaningfully)
        """
        topic = _infer_topic(sents)
        leaf  = LeafCluster(
            depth=depth, topic=topic, sentences=sents,
            similarity=similarity, parent_label=parent_label,
            sub_label=sub_label,
        )

        if run_stance and len(sents) >= MIN_STANCE_SIZE:
            try:
                print(f"    [depth={depth}] running stance split on {len(sents)} sentences …")
                groups = _split_by_stance(sents, device=self.device, confidence=stance_confidence)
                leaf.has_stance_split = True
                leaf.stance_sides = {
                    "positive": groups["positive"],
                    "negative": groups["negative"],
                    "neutral" : groups["neutral"],
                }
                n_pos = len(groups["positive"])
                n_neg = len(groups["negative"])
                n_neu = len(groups["neutral"])
                print(
                    f"      → positive={n_pos}  negative={n_neg}  neutral={n_neu}"
                )
            except Exception as e:
                print(f"    [depth={depth}] stance split failed ({e}), skipping")

        return leaf

    def _leaf_from_indices(
        self,
        indices          : list[int],
        depth            : int,
        parent_label     : int,
        sub_label        : int,
        similarity       : float,
        run_stance       : bool,
        stance_confidence: float,
        topic_override   : Optional[str] = None,
    ) -> LeafCluster:
        sents = [self.sentences[i] for i in indices]
        leaf = self._make_leaf(
            indices, sents, depth, parent_label, sub_label,
            similarity, run_stance, stance_confidence,
        )
        if topic_override is not None:
            leaf.topic = topic_override
        return leaf

    # ── Search quality metrics ─────────────────────────────────────────────

    def _compute_search_metrics(self, results: list[LeafCluster]) -> dict:
        """Compute quality metrics for search results."""
        if not results:
            return {}
        
        # Similarity percentile: where does top result rank?
        top_sim = results[0].similarity
        all_cluster_sims = []
        for lbl_str in self.combined_topics.keys():
            try:
                idx = self.label_to_indices.get(lbl_str, [])
                if idx:
                    # Quick approximate: centroid similarity
                    all_cluster_sims.append(top_sim)
            except:
                pass
        
        if all_cluster_sims:
            percentile = 100 * np.mean(np.array(all_cluster_sims) <= top_sim)
        else:
            percentile = 50.0
        
        # Coverage: how many docs are in results?
        total_docs = sum(len(r.sentences) for r in results)
        coverage_pct = 100 * total_docs / len(self.sentences)
        
        # Diversity: how many different top-level clusters are represented?
        clusters_represented = len(set(r.parent_label for r in results))
        
        return {
            "similarity_percentile": percentile,
            "coverage_pct": coverage_pct,
            "clusters_represented": clusters_represented,
            "total_docs": total_docs,
        }

    # ── Public search API ──────────────────────────────────────────────

    def search(
        self,
        query            : str,
        top_k            : int   = 5,
        min_similarity   : float = 0.2,
        max_depth        : int   = DEFAULT_MAX_DEPTH,
        min_leaf_size    : int   = 10,
        refine           : bool  = True,
        detect_stance    : bool  = True,
        stance_confidence: float = 0.60,
        top_k_per_level  : int   = 3,   # ← new: keep only top-K sub-clusters at each HDBSCAN level
        use_faiss        : bool  = True, # use FAISS IVF + rerank for final retrieval
        final_candidate_docs: int = 5000,
        faiss_top_n      : int  = 200,
        exact_rerank_top_n: int = 50,
        nli_final_docs   : int  = 20,
        nprobe           : int  = 10,
    ) -> list[LeafCluster]:
        """
        Find semantically relevant clusters and split by news framing.

        Parameters
        ----------
        query             : natural-language search string
        top_k             : number of top-level clusters to retrieve
        min_similarity    : cosine similarity threshold (0–1)
        max_depth         : max query-time recursion depth after the two precomputed layers
        min_leaf_size     : clusters smaller than this are returned without recursing
        refine            : False = skip query-time refinement and return layer-2 clusters
        detect_stance     : True = run stance detection at leaf clusters
        stance_confidence : NLI confidence threshold for assigning a stance label
                            (sentences below this go to 'neutral')

        Returns
        -------
        List of LeafCluster objects sorted by similarity (desc).
        Clusters with has_stance_split=True have .stance_sides populated.
        """
        print(f"\nQuery: '{query}'")

        # ── Stage 1: retrieve top-K precomputed layer-1 groups ─────────────
        query_vec = self._embed_query(query)
        group_hits = self._retrieve_clusters(
            query_vec,
            top_k=top_k,
            min_similarity=min_similarity,
            centroid_ids=self._group_centroid_ids,
            centroid_mat=self._group_centroid_mat,
        )

        if not group_hits:
            print("  No clusters found.")
            return []

        # ── Stage 2: retrieve top-K precomputed layer-2 clusters per group ──
        selected_cluster_hits: list[tuple[int, float, int]] = []
        for group_id, group_sim in group_hits:
            cluster_ids = self._group_to_cluster_ids.get(int(group_id), [])
            if not cluster_ids:
                continue

            cluster_pos = [self._centroid_index[cid] for cid in cluster_ids if cid in self._centroid_index]
            if not cluster_pos:
                continue

            child_hits = self._retrieve_clusters(
                query_vec=query_vec,
                top_k=top_k_per_level,
                min_similarity=min_similarity,
                centroid_ids=self._centroid_ids[cluster_pos],
                centroid_mat=self._centroid_mat[cluster_pos],
            )
            for child_id, child_sim in child_hits:
                selected_cluster_hits.append((child_id, child_sim, int(group_id)))

        # ── Document-level retrieval via FAISS (coarse routing by centroids) ──
        if use_faiss:
            candidate_labels = [int(lbl) for lbl, _, _ in selected_cluster_hits]
            doc_ids, approx_scores, doc_labels = self._faiss_search_labels(
                query_vec=query_vec,
                labels=candidate_labels,
                faiss_top_n=faiss_top_n,
                nprobe=nprobe,
            )

            if doc_ids.size > 0:
                # aggregate by doc id (keep best approx score)
                order = np.argsort(approx_scores)[::-1]
                doc_ids = doc_ids[order]
                approx_scores = approx_scores[order]

                # keep up to final_candidate_docs unique docs
                unique_map = {}
                uniq_list = []
                uniq_scores = []
                for did, sc, lbl in zip(doc_ids, approx_scores, doc_labels):
                    did_int = int(did)
                    if did_int in unique_map:
                        continue
                    unique_map[did_int] = int(lbl)
                    uniq_list.append(did_int)
                    uniq_scores.append(float(sc))
                    if len(uniq_list) >= final_candidate_docs:
                        break

                if uniq_list:
                    cand_ids = np.array(uniq_list, dtype=np.int64)
                    cand_embs = self.embeddings[cand_ids]
                    exact_scores = (cand_embs @ query_vec).astype(np.float32)
                    top_idx = np.argsort(exact_scores)[::-1][:exact_rerank_top_n]
                    final_ids = cand_ids[top_idx]
                    final_scores = exact_scores[top_idx]

                    # Optional: run NLI/stance on returned docs
                    nli_results = {}
                    if detect_stance and nli_final_docs > 0:
                        nli_ids = final_ids[:nli_final_docs]
                        nli_sents = [self.sentences[int(i)] for i in nli_ids]
                        try:
                            scores = _StanceDetector.get(device=self.device).score(nli_sents)
                            for nli_id, score_dict in zip(nli_ids, scores):
                                nli_results[int(nli_id)] = score_dict
                        except Exception:
                            pass  # NLI failed, skip

                    # Build LeafCluster objects for final docs (one sentence each)
                    doc_leaves: list[LeafCluster] = []
                    for did, sc in zip(final_ids.tolist(), final_scores.tolist()):
                        did_int = int(did)
                        s = self.sentences[did_int]
                        chosen_lbl = unique_map.get(did_int, int(self.combined_labels[did_int]))
                        topic = self.combined_topics.get(str(chosen_lbl), _infer_topic([s]))
                        
                        # Attach NLI result if available
                        nli_info = nli_results.get(did_int, {})
                        
                        leaf = LeafCluster(
                            depth=0,
                            topic=topic,
                            sentences=[s],
                            similarity=float(sc),
                            parent_label=chosen_lbl,
                            sub_label=-1,
                            nli_label=nli_info.get("label"),
                            nli_score=nli_info.get("score", 0.0),
                        )
                        doc_leaves.append(leaf)

                    return doc_leaves
        # ── No refinement: return precomputed layer-2 clusters (+ stance) ──
        if not refine or max_depth == 0:
            leaves = []
            for lbl, sim, group_id in selected_cluster_hits:
                idx = self.label_to_indices.get(str(lbl), [])
                topic = self.combined_topics.get(
                    str(lbl), _infer_topic([self.sentences[i] for i in idx])
                )
                leaves.append(self._leaf_from_indices(
                    indices=idx,
                    depth=0,
                    parent_label=group_id,
                    sub_label=-1,
                    similarity=sim,
                    run_stance=detect_stance,
                    stance_confidence=stance_confidence,
                    topic_override=topic,
                ))
            return sorted(leaves, key=lambda x: -x.similarity)

        # ── Stage 3: query-time refinement with early exit ────────────────
        all_leaves: list[LeafCluster] = []

        for lbl, sim, group_id in selected_cluster_hits:
            idx = self.label_to_indices.get(str(lbl), [])
            if not idx:
                continue
            topic = self.combined_topics.get(str(lbl), f"cluster {lbl}")

            leaves = self._recursive_refine(
                indices          = idx,
                query_vec        = query_vec,
                parent_label     = group_id,
                parent_sim       = sim,
                depth            = 0,
                max_depth        = max_depth,
                min_leaf_size    = min_leaf_size,
                run_stance       = detect_stance,
                stance_confidence= stance_confidence,
                top_k_per_level  = top_k_per_level,   # ← new
            )
            for lf in leaves:
                stance_note = " [bias-split]" if lf.has_stance_split else ""
                print(
                    f"    depth={lf.depth}  sim={lf.similarity:.3f}  "
                    f"n={len(lf.sentences):4d}  {lf.topic[:50]}{stance_note}"
                )
            all_leaves.extend(leaves)

        all_leaves.sort(key=lambda x: -x.similarity)

        # Enforce top-k results overall (refinement can create > top_k leaves).
        if len(all_leaves) > top_k:
            all_leaves = all_leaves[:top_k]

        bias_count = sum(1 for l in all_leaves if l.has_stance_split)
        metrics = self._compute_search_metrics(all_leaves)
        print(f"Returned {len(all_leaves)} results (stance split: {bias_count})")
        if metrics:
            print(f"  Coverage: {metrics['coverage_pct']:.1f}% | Diversity: {metrics['clusters_represented']} clusters")
        return all_leaves

    # ── Display ────────────────────────────────────────────────────────────

    def display_results(
        self,
        results   : list[LeafCluster],
        max_sents : int = 5,
    ) -> None:
        """Pretty-print search results with quality metrics."""
        if not results:
            print("No results.")
            return

        # Calculate per-result metrics
        max_sim = max((r.similarity for r in results), default=0)
        total_results_docs = sum(len(r.sentences) for r in results)

        for i, r in enumerate(results, 1):
            # Similarity percentile within this result set
            sim_pct = 100 * r.similarity / max_sim if max_sim > 0 else 0
            
            # Result header with topic, similarity, and quality info
            print(f"\n{i}. [{r.topic[:50]}]")
            print(f"   Sim: {r.similarity:.3f} ({sim_pct:.0f}% of top) | Docs: {len(r.sentences)}")
            
            # Show NLI stance if available
            if r.nli_label:
                label_short = "POS" if "positive" in r.nli_label else "NEG" if "negative" in r.nli_label else "?"
                print(f"   Stance: {label_short} (conf={r.nli_score:.2f})")
            
            # Show sample sentences
            for s in r.sentences[:max_sents]:
                print(f"   • {s}")
            if len(r.sentences) > max_sents:
                print(f"   … and {len(r.sentences) - max_sents} more")

            # Show bias split if detected
            if r.has_stance_split:
                print(f"   [BIAS SPLIT]")
                for side_name in ["positive", "negative", "neutral"]:
                    side_sents = r.stance_sides.get(side_name, [])
                    if side_sents:
                        print(f"     {side_name}: {len(side_sents)} headlines")
