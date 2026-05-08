"""
cluster_search.py
─────────────────
Two-stage semantic search over saved cluster data, with bias detection.

  Stage 1 — Cosine similarity search
      Encode the user query with the same BGE model used during clustering.
      Rank all cluster centroids by cosine similarity → return top-K clusters.

  Stage 2 — Shallow HDBSCAN refinement (max 2 levels)
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
from pathlib import Path
from typing import Optional

import numpy as np

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

# Max depth for HDBSCAN recursion (reduced from 3 → 2).
DEFAULT_MAX_DEPTH = 2


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
    """
    depth            : int
    topic            : str
    sentences        : list[str]
    similarity       : float
    parent_label     : int
    sub_label        : int          = -1
    has_stance_split : bool         = False
    stance_sides     : dict         = field(default_factory=dict)
    # stance_sides = {
    #   "positive": [sentences favouring positive framing],
    #   "negative": [sentences favouring negative framing],
    #   "neutral" : [sentences without clear stance],
    # }

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
    """PCA → UMAP reduction. Returns float32 array."""
    from sklearn.decomposition import PCA
    from umap import UMAP

    N = embeddings.shape[0]
    if N < 4:
        return embeddings.astype(np.float32)

    pca_dims = min(150, N - 1, embeddings.shape[1])
    pca_out  = PCA(n_components=pca_dims).fit_transform(embeddings)

    umap_dims = min(target_dims, pca_out.shape[1], N - 1)
    n_neigh   = max(2, min(30, N - 1))
    reduced   = UMAP(
        n_components=umap_dims,
        n_neighbors=n_neigh,
        min_dist=0.0,
        metric="euclidean",
        random_state=42,
        low_memory=True,
        n_epochs=300,
    ).fit_transform(pca_out)

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
        self._stance        = None   # stance detector, lazy-loaded on first use
        self._load_store()

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
        self.embeddings      : np.ndarray     = npz["embeddings"]
        self.group_labels    : np.ndarray     = npz["group_labels"]
        self.sub_labels      : np.ndarray     = npz["sub_labels"]
        self.combined_labels : np.ndarray     = npz["combined_labels"]

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.sentences       : list[str]      = meta["sentences"]
        self.group_topics    : list[str]      = meta["group_topics"]
        self.combined_topics : dict[str, str] = meta["combined_topics"]
        self.label_to_indices: dict[str, list[int]] = meta["label_to_indices"]

        cids, cvecs = [], []
        for k, v in meta["cluster_centroids"].items():
            cids.append(int(k))
            cvecs.append(v)

        self._centroid_ids : np.ndarray = np.array(cids,  dtype=np.int64)
        self._centroid_mat : np.ndarray = np.array(cvecs, dtype=np.float32)

        print(
            f"  ✓ {len(self.sentences):,} sentences  |  "
            f"{meta['n_groups']} groups  |  "
            f"{meta['n_clusters']} clusters  |  "
            f"embeddings {self.embeddings.shape}"
        )

    # ── Embedding the query ────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """Encode query; returns unit-norm float32 vector shape (D,)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading encoder '{self.model_name}' on {self.device} …")
            self._model = SentenceTransformer(self.model_name, device=self.device)

        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        vec = self._model.encode(
            [prefixed],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return vec.astype(np.float32)

    # ── Stage 1: cosine similarity over cluster centroids ─────────────────

    def _retrieve_clusters(
        self, query_vec: np.ndarray, top_k: int, min_similarity: float
    ) -> list[tuple[int, float]]:
        sims  = self._centroid_mat @ query_vec
        order = np.argsort(sims)[::-1]
        results = []
        for idx in order[:top_k]:
            sim = float(sims[idx])
            if sim < min_similarity:
                break
            results.append((int(self._centroid_ids[idx]), sim))
        return results

    # ── Stage 2: shallow HDBSCAN with early exit ───────────────────────────

    def _recursive_refine(
        self,
        indices         : list[int],
        query_vec       : np.ndarray,
        parent_label    : int,
        parent_sim      : float,
        depth           : int,
        max_depth       : int,
        min_leaf_size   : int,
        run_stance      : bool,
        stance_confidence: float,
    ) -> list[LeafCluster]:
        """
        Recursively refine a cluster with early-exit logic.

        Decision tree at each node:
          1. Too small (< min_leaf_size)     → leaf immediately
          2. Depth limit reached             → leaf, then maybe stance split
          3. Already homogeneous             → skip HDBSCAN, go to stance split
          4. HDBSCAN finds ≤ 1 real cluster  → leaf, then maybe stance split
          5. HDBSCAN splits cleanly         → recurse into each sub-cluster
        """
        sents = [self.sentences[i] for i in indices]
        M     = len(indices)
        embs  = self.embeddings[np.array(indices)]

        # ── 1. Too small ───────────────────────────────────────────────────
        if M < 4:
            return [self._make_leaf(
                indices, sents, embs, depth, parent_label, -1,
                parent_sim, run_stance, stance_confidence,
            )]

        # ── 2. Depth limit reached ─────────────────────────────────────────
        if depth >= max_depth:
            print(f"    [depth={depth}] depth limit — running stance split")
            return [self._make_leaf(
                indices, sents, embs, depth, parent_label, -1,
                parent_sim, run_stance, stance_confidence,
            )]

        # ── 3. Homogeneity check — skip HDBSCAN if already tight ──────────
        if _is_homogeneous(embs, threshold=HOMOGENEITY_THRESHOLD):
            print(
                f"    [depth={depth}] cluster homogeneous (n={M}) "
                f"— skipping HDBSCAN, running stance split"
            )
            return [self._make_leaf(
                indices, sents, embs, depth, parent_label, -1,
                parent_sim, run_stance, stance_confidence,
            )]

        # ── 4. Run HDBSCAN ─────────────────────────────────────────────────
        try:
            reduced = _reduce(embs, target_dims=min(50, M - 1))
            labels  = _run_hdbscan(reduced, group_size=M)
        except Exception as e:
            print(f"    [depth={depth}] HDBSCAN failed ({e}), returning as leaf")
            return [self._make_leaf(
                indices, sents, embs, depth, parent_label, -1,
                parent_sim, run_stance, stance_confidence,
            )]

        unique_subs = sorted(set(labels) - {-1})

        # ── 5a. No real sub-clusters → leaf + stance ───────────────────────
        if len(unique_subs) <= 1:
            print(f"    [depth={depth}] HDBSCAN found no split (n={M}) — running stance split")
            return [self._make_leaf(
                indices, sents, embs, depth, parent_label, -1,
                parent_sim, run_stance, stance_confidence,
            )]

        # ── 5b. Recurse into sub-clusters ─────────────────────────────────
        leaves: list[LeafCluster] = []

        # Noise points → immediate leaf (no further recursion, no stance split
        # because they are by definition the points that don't belong)
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

        for sub_id in unique_subs:
            sub_idx = [indices[i] for i, l in enumerate(labels) if l == sub_id]
            sub_embs = self.embeddings[np.array(sub_idx)]

            # Sub-cluster similarity to query
            sub_centroid = sub_embs.mean(axis=0)
            norm = np.linalg.norm(sub_centroid)
            if norm > 1e-9:
                sub_centroid /= norm
            sub_sim = float(sub_centroid @ query_vec)

            # Too small for further recursion
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

            # Recurse
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
            )
            leaves.extend(children)

        return leaves

    # ── Leaf builder (applies stance detection when appropriate) ───────────

    def _make_leaf(
        self,
        indices          : list[int],
        sents            : list[str],
        embs             : np.ndarray,
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

    # ── Public search API ──────────────────────────────────────────────────

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
    ) -> list[LeafCluster]:
        """
        Find semantically relevant clusters and split by news framing.

        Parameters
        ----------
        query             : natural-language search string
        top_k             : number of top-level clusters to retrieve
        min_similarity    : cosine similarity threshold (0–1)
        max_depth         : max HDBSCAN recursion depth (default 2, was 3)
        min_leaf_size     : clusters smaller than this are returned without recursing
        refine            : False = skip HDBSCAN entirely (fastest, coarsest)
        detect_stance     : True = run stance detection at leaf clusters
        stance_confidence : NLI confidence threshold for assigning a stance label
                            (sentences below this go to 'neutral')

        Returns
        -------
        List of LeafCluster objects sorted by similarity (desc).
        Clusters with has_stance_split=True have .stance_sides populated.
        """
        print(f"\n{'='*60}")
        print(f"Query : '{query}'")
        print(f"{'='*60}")

        # ── Stage 1: retrieve top-K clusters by cosine similarity ──────────
        query_vec = self._embed_query(query)
        hits      = self._retrieve_clusters(query_vec, top_k=top_k,
                                            min_similarity=min_similarity)

        if not hits:
            print("No clusters found above similarity threshold.")
            return []

        print(f"\nStage 1 — top {len(hits)} clusters by cosine similarity:")
        for lbl, sim in hits:
            topic = self.combined_topics.get(str(lbl), f"cluster {lbl}")
            count = len(self.label_to_indices.get(str(lbl), []))
            print(f"  [{lbl}]  sim={sim:.3f}  n={count:4d}  {topic[:60]}")

        # ── No refinement: return top-level clusters (+ stance if requested) ──
        if not refine or max_depth == 0:
            leaves = []
            for lbl, sim in hits:
                idx   = self.label_to_indices.get(str(lbl), [])
                sents = [self.sentences[i] for i in idx]
                embs  = self.embeddings[np.array(idx)] if idx else np.empty((0,))
                topic = self.combined_topics.get(str(lbl), _infer_topic(sents))
                leaf  = self._make_leaf(
                    indices=idx, sents=sents, embs=embs,
                    depth=0, parent_label=lbl, sub_label=-1,
                    similarity=sim, run_stance=detect_stance,
                    stance_confidence=stance_confidence,
                )
                leaf.topic = topic
                leaves.append(leaf)
            return sorted(leaves, key=lambda x: -x.similarity)

        # ── Stage 2: shallow HDBSCAN refinement with early exit ───────────
        print(
            f"\nStage 2 — HDBSCAN refinement "
            f"(max_depth={max_depth}, homogeneity_threshold={HOMOGENEITY_THRESHOLD}) …"
        )
        all_leaves: list[LeafCluster] = []

        for lbl, sim in hits:
            idx = self.label_to_indices.get(str(lbl), [])
            if not idx:
                continue
            topic = self.combined_topics.get(str(lbl), f"cluster {lbl}")
            print(f"\n  Refining cluster [{lbl}]  '{topic[:50]}'  ({len(idx)} pts)")

            leaves = self._recursive_refine(
                indices          = idx,
                query_vec        = query_vec,
                parent_label     = lbl,
                parent_sim       = sim,
                depth            = 0,
                max_depth        = max_depth,
                min_leaf_size    = min_leaf_size,
                run_stance       = detect_stance,
                stance_confidence= stance_confidence,
            )
            for lf in leaves:
                stance_note = " [bias-split]" if lf.has_stance_split else ""
                print(
                    f"    depth={lf.depth}  sim={lf.similarity:.3f}  "
                    f"n={len(lf.sentences):4d}  {lf.topic[:50]}{stance_note}"
                )
            all_leaves.extend(leaves)

        all_leaves.sort(key=lambda x: -x.similarity)

        print(f"\n{'='*60}")
        print(f"Returned {len(all_leaves)} leaf clusters")
        bias_count = sum(1 for l in all_leaves if l.has_stance_split)
        print(f"  of which {bias_count} received a stance split")
        print(f"{'='*60}\n")
        return all_leaves

    # ── Display ────────────────────────────────────────────────────────────

    def display_results(
        self,
        results   : list[LeafCluster],
        max_sents : int = 5,
    ) -> None:
        """Pretty-print search results, showing stance splits when present."""
        if not results:
            print("No results.")
            return

        for i, r in enumerate(results, 1):
            print(f"\n{'─'*60}")
            print(f"Result {i}  |  depth={r.depth}  sim={r.similarity:.3f}  n={len(r.sentences)}")
            print(f"Topic   : {r.topic}")
            print(f"Parent  : cluster {r.parent_label}")

            if r.has_stance_split:
                # Show both sides of the framing split
                print(f"\n  ◈ BIAS SPLIT DETECTED")
                sides = [
                    ("positive framing", r.stance_sides.get("positive", [])),
                    ("negative framing", r.stance_sides.get("negative", [])),
                    ("neutral",          r.stance_sides.get("neutral",  [])),
                ]
                for side_name, side_sents in sides:
                    if not side_sents:
                        continue
                    print(f"\n  [{side_name.upper()}]  ({len(side_sents)} headlines)")
                    for s in side_sents[:max_sents]:
                        print(f"    • {s}")
                    if len(side_sents) > max_sents:
                        print(f"    … and {len(side_sents) - max_sents} more")
            else:
                print(f"Samples :")
                for s in r.sentences[:max_sents]:
                    print(f"    • {s}")
                if len(r.sentences) > max_sents:
                    print(f"    … and {len(r.sentences) - max_sents} more")

        print(f"\n{'─'*60}")
