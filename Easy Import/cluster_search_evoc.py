"""
cluster_search_evoc.py
──────────────────────
Hierarchical semantic search over cached EVōC results.

This backend is meant for Colab_Main_with_EVoC.ipynb.
That notebook precomputes the EVōC hierarchy and stores a results cache
containing embeddings, sentences, cluster_layers_, and topic metadata.

Search flow
-----------
1. Rank the coarsest EVōC layer by cosine similarity.
2. Drill into the best children on the next EVōC layer.
3. Repeat until the requested depth is reached, the hierarchy ends,
   or a cluster is already semantically tight.
4. Run stance/NLI only at the final leaf clusters.
"""

from __future__ import annotations

import csv
import hashlib
import html
import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Tunable constants
# ──────────────────────────────────────────────────────────────────────────────

HOMOGENEITY_THRESHOLD = 0.08
MIN_STANCE_SIZE = 15
STANCE_MODEL = "cross-encoder/nli-deberta-v3-small"
STANCE_LABELS = ["positive coverage", "negative coverage"]
DEFAULT_MAX_DEPTH = 6


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LeafCluster:
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


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

_RE_SOURCE_SUFFIX = re.compile(r"\s+-\s+\S+\.\S+$")
_RE_HTML_TAG      = re.compile(r"<[^>]+>")
_RE_NONALPHA      = re.compile(r"[^a-zA-Z0-9\s\'\-]")
_RE_SPACES        = re.compile(r"\s+")
_RE_SHORT_NUM     = re.compile(r"(?<![a-zA-Z0-9\-])\d{1,3}(?![a-zA-Z0-9\-])")
_RE_THOUSANDS     = re.compile(r"(\d),(\d)")


def clean_title(raw: str) -> str:
    s = html.unescape(raw)
    s = html.unescape(s)
    s = _RE_SOURCE_SUFFIX.sub("", s)
    s = _RE_HTML_TAG.sub(" ", s)
    s = _RE_THOUSANDS.sub(r"\1\2", s)
    s = _RE_THOUSANDS.sub(r"\1\2", s)
    s = _RE_NONALPHA.sub(" ", s)
    s = _RE_SHORT_NUM.sub(" ", s)
    s = _RE_SPACES.sub(" ", s)
    return s.strip()


def load_titles_from_csv(csv_file: Path) -> list[str]:
    titles: list[str] = []
    with csv_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("Title") or "").strip()
            if not raw:
                continue
            cleaned = clean_title(raw)
            if cleaned:
                titles.append(cleaned)
    return titles


def data_fingerprint(sentences: list[str]) -> str:
    return hashlib.md5("\n".join(sentences).encode()).hexdigest()[:8]


def _resolve_cache_dir() -> str:
    drive_dir = "/content/drive/MyDrive"
    if os.path.isdir(drive_dir):
        cache_dir = os.path.join(drive_dir, "clustering_cache_evoc")
    else:
        cache_dir = "/tmp/clustering_cache_evoc"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _cache_path(cache_dir: str, name: str) -> str:
    return os.path.join(cache_dir, f"{name}.pkl")


def load_cache(cache_dir: str, name: str):
    path = _cache_path(cache_dir, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Topic and stance helpers
# ──────────────────────────────────────────────────────────────────────────────


def _infer_topic(sentences: list[str], top_terms: int = 5) -> str:
    from collections import Counter
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not sentences:
        return "miscellaneous"

    try:
        analyzer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).build_analyzer()
        counts: Counter = Counter()
        for sent in sentences:
            counts.update(analyzer(sent))
        if not counts:
            return "miscellaneous"
        return " ".join(term for term, _ in counts.most_common(top_terms))
    except Exception:
        return "miscellaneous"


class _StanceDetector:
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
        if not sentences:
            return []

        results = self._pipe(
            sentences,
            candidate_labels=STANCE_LABELS,
            multi_label=False,
            batch_size=32,
        )

        out = []
        for result in results:
            out.append({"label": result["labels"][0], "score": result["scores"][0]})
        return out


def _split_by_stance(sentences: list[str], device: str = "cpu", confidence: float = 0.60) -> dict:
    detector = _StanceDetector.get(device=device)
    scored = detector.score(sentences)
    groups: dict = {"positive": [], "negative": [], "neutral": [], "scores": scored}

    for sent, result in zip(sentences, scored):
        if result["score"] >= confidence:
            key = "positive" if result["label"] == "positive coverage" else "negative"
        else:
            key = "neutral"
        groups[key].append(sent)

    return groups


# ──────────────────────────────────────────────────────────────────────────────
# Search backend
# ──────────────────────────────────────────────────────────────────────────────


class EvocClusterSearch:
    """Search EVōC layers from coarsest to finest and run stance only at leaves."""

    def __init__(
        self,
        cache_dir    : str           = "",
        csv_file     : Path | None   = None,
        model_name   : str           = "BAAI/bge-large-en-v1.5",
        device       : Optional[str] = None,
        stance_model : str           = STANCE_MODEL,
        results_key  : str | None    = None,
    ) -> None:
        import torch

        self.cache_dir = cache_dir or _resolve_cache_dir()
        self.csv_file = csv_file or Path("/content/train.csv")
        self.model_name = model_name
        self.stance_model = stance_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._load_store(results_key=results_key)

    def _results_key(self) -> str:
        fp = data_fingerprint(load_titles_from_csv(self.csv_file))
        return f"evoc_results_{fp}"

    def _load_store(self, results_key: str | None = None) -> None:
        key = results_key or self._results_key()
        payload = load_cache(self.cache_dir, key)
        if payload is None:
            raise FileNotFoundError(
                f"EVōC results cache not found for '{key}' in '{self.cache_dir}'.\n"
                "Run Colab_Main_with_EVoC.ipynb first so it writes evoc_results_<fingerprint>.pkl."
            )

        self.sentences: list[str] = payload["sentences"]
        self.embeddings: np.ndarray = np.asarray(payload["embeddings"], dtype=np.float32)
        self.original_layers: list[np.ndarray] = [np.asarray(layer, dtype=np.int32) for layer in payload["cluster_layers"]]
        self.topic_map: dict[int, str] = {int(k): v for k, v in payload.get("topics_dict", {}).items()}
        self.duplicates = payload.get("duplicates", [])

        if not self.original_layers:
            raise ValueError("EVōC results do not contain any cluster layers.")

        # Search from coarsest to finest, which matches a query-time drill-down flow.
        self.search_layers: list[np.ndarray] = list(reversed(self.original_layers))
        self.layer_count = len(self.search_layers)

        self.layer_label_to_indices: list[dict[int, list[int]]] = []
        self.layer_centroids: list[dict[int, np.ndarray]] = []
        self.layer_topics: list[dict[int, str]] = []
        self.parent_children: list[dict[int, list[int]]] = []

        for level_idx, labels in enumerate(self.search_layers):
            label_to_indices: dict[int, list[int]] = {}
            for doc_idx, label in enumerate(labels.tolist()):
                if int(label) == -1:
                    continue
                label_to_indices.setdefault(int(label), []).append(doc_idx)
            self.layer_label_to_indices.append(label_to_indices)

            centroids: dict[int, np.ndarray] = {}
            topics: dict[int, str] = {}
            for label, doc_ids in label_to_indices.items():
                vec = self.embeddings[np.array(doc_ids)].mean(axis=0)
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec = vec / norm
                centroids[label] = vec.astype(np.float32)

                layer_sentences = [self.sentences[i] for i in doc_ids]
                if level_idx == self.layer_count - 1 and label in self.topic_map:
                    topics[label] = self.topic_map[label]
                else:
                    topics[label] = _infer_topic(layer_sentences)

            self.layer_centroids.append(centroids)
            self.layer_topics.append(topics)

            if level_idx < self.layer_count - 1:
                parent_to_children: dict[int, set[int]] = {}
                parent_labels = self.search_layers[level_idx]
                child_labels = self.search_layers[level_idx + 1]
                for parent, child in zip(parent_labels.tolist(), child_labels.tolist()):
                    if int(parent) == -1 or int(child) == -1:
                        continue
                    parent_to_children.setdefault(int(parent), set()).add(int(child))
                self.parent_children.append({k: sorted(v) for k, v in parent_to_children.items()})

        print(f"Loaded EVōC search store from {self.cache_dir}")
        print(f"  {len(self.sentences):,} sentences | {self.layer_count} hierarchy layers")

    def _embed_query(self, query: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)

        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        vec = self._model.encode([prefixed], convert_to_numpy=True, normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def _is_homogeneous(self, indices: list[int]) -> bool:
        if len(indices) < 4:
            return True

        embs = self.embeddings[np.array(indices)]
        centroid = embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm < 1e-9:
            return False
        centroid = centroid / norm
        avg_sim = float(np.mean(embs @ centroid))
        return avg_sim >= 1.0 - HOMOGENEITY_THRESHOLD

    def _make_leaf(
        self,
        level_idx        : int,
        label            : int,
        parent_label     : int,
        similarity       : float,
        run_stance       : bool,
        stance_confidence: float,
    ) -> LeafCluster:
        indices = self.layer_label_to_indices[level_idx].get(label, [])
        sents = [self.sentences[i] for i in indices]
        topic = self.layer_topics[level_idx].get(label, _infer_topic(sents))

        leaf = LeafCluster(
            depth=level_idx,
            topic=topic,
            sentences=sents,
            similarity=similarity,
            parent_label=parent_label,
            sub_label=label,
        )

        if run_stance and len(sents) >= MIN_STANCE_SIZE:
            try:
                print(f"    [layer={level_idx}] running stance split on {len(sents)} sentences …")
                groups = _split_by_stance(sents, device=self.device, confidence=stance_confidence)
                leaf.has_stance_split = True
                leaf.stance_sides = {
                    "positive": groups["positive"],
                    "negative": groups["negative"],
                    "neutral" : groups["neutral"],
                }
            except Exception as exc:
                print(f"    [layer={level_idx}] stance split failed ({exc}), skipping")

        return leaf

    def _walk_layer(
        self,
        level_idx        : int,
        label            : int,
        query_vec        : np.ndarray,
        depth            : int,
        max_depth        : int,
        min_leaf_size    : int,
        run_stance       : bool,
        stance_confidence: float,
        top_k_per_level  : int,
        parent_label     : int,
    ) -> list[LeafCluster]:
        indices = self.layer_label_to_indices[level_idx].get(label, [])
        if not indices:
            return []

        similarity = float(self.layer_centroids[level_idx][label] @ query_vec)

        if depth >= max_depth:
            return [self._make_leaf(level_idx, label, parent_label, similarity, run_stance, stance_confidence)]

        if level_idx >= self.layer_count - 1:
            return [self._make_leaf(level_idx, label, parent_label, similarity, run_stance, stance_confidence)]

        if len(indices) < min_leaf_size or self._is_homogeneous(indices):
            return [self._make_leaf(level_idx, label, parent_label, similarity, run_stance, stance_confidence)]

        child_labels = self.parent_children[level_idx].get(label, [])
        if not child_labels:
            return [self._make_leaf(level_idx, label, parent_label, similarity, run_stance, stance_confidence)]

        scored_children: list[tuple[int, float]] = []
        child_centroids = self.layer_centroids[level_idx + 1]
        for child_label in child_labels:
            child_vec = child_centroids.get(child_label)
            if child_vec is None:
                continue
            child_sim = float(child_vec @ query_vec)
            scored_children.append((child_label, child_sim))

        if not scored_children:
            return [self._make_leaf(level_idx, label, parent_label, similarity, run_stance, stance_confidence)]

        scored_children.sort(key=lambda item: -item[1])
        kept_children = scored_children[:top_k_per_level]

        if len(scored_children) > len(kept_children):
            kept_str = ", ".join(f"{child}:{score:.3f}" for child, score in kept_children)
            print(f"    [layer={level_idx}] pruned {len(scored_children) - len(kept_children)} children: {kept_str}")

        leaves: list[LeafCluster] = []
        for child_label, child_sim in kept_children:
            leaves.extend(self._walk_layer(
                level_idx=level_idx + 1,
                label=child_label,
                query_vec=query_vec,
                depth=depth + 1,
                max_depth=max_depth,
                min_leaf_size=min_leaf_size,
                run_stance=run_stance,
                stance_confidence=stance_confidence,
                top_k_per_level=top_k_per_level,
                parent_label=label,
            ))

        return leaves

    def search(
        self,
        query            : str,
        top_k            : int   = 5,
        min_similarity   : float = 0.2,
        max_depth        : int   = DEFAULT_MAX_DEPTH,
        min_leaf_size    : int   = 10,
        detect_stance    : bool  = True,
        stance_confidence: float = 0.60,
        top_k_per_level  : int   = 3,
    ) -> list[LeafCluster]:
        print(f"\nQuery: '{query}'")

        query_vec = self._embed_query(query)
        top_level_labels = self._retrieve_level(query_vec, level_idx=0, top_k=top_k, min_similarity=min_similarity)

        if not top_level_labels:
            print("  No clusters found.")
            return []

        all_leaves: list[LeafCluster] = []
        for label, score in top_level_labels:
            all_leaves.extend(self._walk_layer(
                level_idx=0,
                label=label,
                query_vec=query_vec,
                depth=0,
                max_depth=max_depth,
                min_leaf_size=min_leaf_size,
                run_stance=detect_stance,
                stance_confidence=stance_confidence,
                top_k_per_level=top_k_per_level,
                parent_label=-1,
            ))

        all_leaves.sort(key=lambda item: -item.similarity)
        if len(all_leaves) > top_k:
            all_leaves = all_leaves[:top_k]

        bias_count = sum(1 for leaf in all_leaves if leaf.has_stance_split)
        print(f"Returned {len(all_leaves)} results (stance split: {bias_count})")
        return all_leaves

    def _retrieve_level(
        self,
        query_vec     : np.ndarray,
        level_idx     : int,
        top_k         : int,
        min_similarity: float,
    ) -> list[tuple[int, float]]:
        centroids = self.layer_centroids[level_idx]
        if not centroids:
            return []

        labels = list(centroids.keys())
        mat = np.vstack([centroids[label] for label in labels]).astype(np.float32)
        sims = mat @ query_vec
        order = np.argsort(sims)[::-1]

        hits: list[tuple[int, float]] = []
        for pos in order[:top_k]:
            score = float(sims[pos])
            if score < min_similarity:
                break
            hits.append((int(labels[pos]), score))
        return hits

    def display_results(self, results: list[LeafCluster], max_sents: int = 5) -> None:
        if not results:
            print("No results.")
            return

        max_sim = max((item.similarity for item in results), default=0.0)

        for idx, result in enumerate(results, 1):
            sim_pct = 100 * result.similarity / max_sim if max_sim > 0 else 0
            print(f"\n{idx}. [{result.topic[:50]}]")
            print(f"   Sim: {result.similarity:.3f} ({sim_pct:.0f}% of top) | Docs: {len(result.sentences)}")

            if result.nli_label:
                label_short = "POS" if "positive" in result.nli_label else "NEG" if "negative" in result.nli_label else "?"
                print(f"   Stance: {label_short} (conf={result.nli_score:.2f})")

            for sent in result.sentences[:max_sents]:
                print(f"   • {sent}")
            if len(result.sentences) > max_sents:
                print(f"   … and {len(result.sentences) - max_sents} more")

            if result.has_stance_split:
                print("   [BIAS SPLIT]")
                for side_name in ["positive", "negative", "neutral"]:
                    side_sents = result.stance_sides.get(side_name, [])
                    if side_sents:
                        print(f"     {side_name}: {len(side_sents)} headlines")


def main() -> None:
    searcher = EvocClusterSearch()
    results = searcher.search("vaccine side effects", top_k=5)
    searcher.display_results(results)


if __name__ == "__main__":
    main()