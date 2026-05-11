#!/usr/bin/env python3
"""Small Flask app to inspect EVōC hierarchies and drill down to leaves.

Run:
    python evoc_dashboard.py --cache-dir /path/to/cache --results-key evoc_results_<fp>

It loads the same EVOC cache used by `Easy Import/cluster_search_evoc.py` and
exposes simple JSON endpoints for an interactive single-page UI.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file

HERE = Path(__file__).parent
EVOC_PY = HERE / "Easy Import" / "cluster_search_evoc.py"
if not EVOC_PY.exists():
    raise FileNotFoundError(f"Cannot find cluster_search_evoc.py at {EVOC_PY}")

spec = importlib.util.spec_from_file_location("cluster_search_evoc", str(EVOC_PY))
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)

EvocClusterSearch = module.EvocClusterSearch
LeafCluster = module.LeafCluster

app = Flask(__name__)
searcher: EvocClusterSearch | None = None


def serialize_leaf(leaf: LeafCluster) -> dict[str, Any]:
    return {
        "depth": int(leaf.depth),
        "topic": leaf.topic,
        "size": len(leaf.sentences),
        "similarity": float(leaf.similarity),
        "parent_label": int(leaf.parent_label),
        "sub_label": int(leaf.sub_label),
        "has_stance_split": bool(leaf.has_stance_split),
        "nli_label": leaf.nli_label,
        "nli_score": float(leaf.nli_score or 0.0),
        "sentences": leaf.sentences,
    }


@app.route("/")
def index():
    return send_file(HERE / "Easy Import" / "evoc_dashboard.html")


@app.route("/api/meta")
def api_meta():
    assert searcher is not None
    layers = []
    for idx, label_to_indices in enumerate(searcher.layer_label_to_indices):
        layer_info = []
        topics = searcher.layer_topics[idx]
        for label, ids in label_to_indices.items():
            layer_info.append({"label": int(label), "size": len(ids), "topic": topics.get(label, "")})
        layers.append(layer_info)

    return jsonify({
        "layer_count": searcher.layer_count,
        "layers": layers,
    })


@app.route("/api/cluster")
def api_cluster():
    assert searcher is not None
    try:
        level = int(request.args["level"])
        label = int(request.args["label"])
    except Exception:
        return jsonify({"error": "provide level and label query params"}), 400

    indices = searcher.layer_label_to_indices[level].get(label, [])
    children = []
    if level < searcher.layer_count - 1:
        children = searcher.parent_children[level].get(label, [])
    topic = searcher.layer_topics[level].get(label, "")
    sample = [searcher.sentences[i] for i in indices[:50]]
    return jsonify({
        "level": level,
        "label": label,
        "size": len(indices),
        "topic": topic,
        "children": children,
        "sample_sentences": sample,
    })


@app.route("/api/search", methods=["POST"])
def api_search():
    assert searcher is not None
    body = request.get_json(force=True)
    query = body.get("query")
    if not query:
        return jsonify({"error": "query required"}), 400
    top_k = int(body.get("top_k", 5))
    max_depth = int(body.get("max_depth", 6))
    results = searcher.search(query, top_k=top_k, max_depth=max_depth, top_k_per_level=int(body.get("top_k_per_level", 3)))
    return jsonify([serialize_leaf(r) for r in results])


def main():
    global searcher
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="", help="EVOC cache dir (optional)")
    parser.add_argument("--results-key", default=None, help="specific results key (evoc_results_<fp>)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    args = parser.parse_args()

    searcher = EvocClusterSearch(cache_dir=args.cache_dir or "", results_key=args.results_key)

    print("Starting Flask server on http://%s:%s" % (args.host, args.port))
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
