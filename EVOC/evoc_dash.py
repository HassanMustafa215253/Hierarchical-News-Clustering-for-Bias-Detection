#!/usr/bin/env python3
"""Plotly Dash app for EVoC drilldown.

Run:
    python evoc_dash.py --cache-dir /path/to/cache --results-key evoc_results_<fp>

This app loads the EVoC cache via `Easy Import/cluster_search_evoc.py`, computes a 2D projection
(PCA or UMAP) for visualization, and shows a scatter where each point is a document.
Selecting points or clusters shows sample sentences and cluster metadata.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import json

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px

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


def compute_2d(embs: np.ndarray) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        proj = reducer.fit_transform(embs)
        return proj
    except Exception:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        return pca.fit_transform(embs)


def build_app(searcher: EvocClusterSearch) -> Dash:
    app = Dash(__name__)
    embs = searcher.embeddings.astype(np.float32)
    proj = compute_2d(embs)

    # searcher.search_layers is coarsest->finest
    layer_count = searcher.layer_count
    layers = searcher.search_layers

    # default level to finest
    default_level = layer_count - 1
    labels = layers[default_level].astype(int)

    df = pd.DataFrame({
        "x": proj[:, 0],
        "y": proj[:, 1],
        "label": labels,
        "doc_idx": np.arange(len(labels)),
    })

    app.layout = html.Div([
        html.H3("EVoC Drilldown (Dash)"),
        html.Div([
            html.Label("Layer:"),
            dcc.Slider(id="layer-slider", min=0, max=layer_count - 1, value=default_level, marks={i: str(i) for i in range(layer_count)}, step=1),
            html.Label("Top K results (search):"),
            dcc.Input(id="top-k", type="number", value=5, min=1, style={"width":"80px"}),
            dcc.Input(id="query-input", placeholder="query string", style={"width":"40%", "marginLeft":"12px"}),
            html.Button("Search", id="search-btn", n_clicks=0),
        ], style={"marginBottom":"12px"}),

        html.Div([
            html.Div(dcc.Graph(id="scatter"), style={"width":"65%","display":"inline-block","verticalAlign":"top"}),
            html.Div(id="panel", style={"width":"33%","display":"inline-block","paddingLeft":"12px","verticalAlign":"top"}),
        ]),

        dcc.Store(id="df-store", data=df.to_json(date_format="iso", orient="split")),
    ], style={"padding":"12px"})


    @app.callback(
        Output("scatter", "figure"),
        Input("layer-slider", "value"),
        State("df-store", "data"),
    )
    def update_scatter(level, df_json):
        df_local = pd.read_json(df_json, orient="split")
        labels = layers[level].astype(int)
        df_local["label"] = labels[df_local["doc_idx"]]
        fig = px.scatter(df_local, x="x", y="y", color=df_local["label"].astype(str), hover_data=["doc_idx"], title=f"Layer {level} view (labels)")
        fig.update_traces(marker_size=6)
        return fig


    @app.callback(
        Output("panel", "children"),
        Input("scatter", "clickData"),
        Input("search-btn", "n_clicks"),
        State("query-input", "value"),
        State("top-k", "value"),
        State("layer-slider", "value"),
    )
    def inspect(clickData, n_clicks, query, top_k, level):
        if clickData and "points" in clickData:
            pt = clickData["points"][0]
            doc_idx = int(pt["customdata"][0]) if "customdata" in pt and pt["customdata"] else int(pt.get("hovertext") or pt.get("text") or pt.get("pointIndex"))
            label = int(layers[level][doc_idx])
            indices = searcher.layer_label_to_indices[level].get(label, [])
            sample = [searcher.sentences[i] for i in indices[:50]]
            topic = searcher.layer_topics[level].get(label, "")
            return [
                html.H4(f"Layer {level} — label {label} — size {len(indices)}"),
                html.Div([html.B("Topic: "), html.Span(topic)]),
                html.H5("Sample documents"),
                html.Div([html.Pre(s) for s in sample]),
            ]

        if n_clicks and query:
            # run search
            results = searcher.search(query, top_k=int(top_k or 5), max_depth=6)
            items = []
            for r in results:
                items.append(html.Div([
                    html.H4(r.topic),
                    html.Div(f"size={len(r.sentences)} sim={r.similarity:.3f}"),
                    html.Button("Inspect Cluster", id={"type":"inspect-btn","index":f"{r.depth}-{r.sub_label}"}),
                    html.Div([html.Pre(s) for s in r.sentences[:20]]),
                ], style={"marginBottom":"12px"}))
            return items

        return html.Div("Click a point or run a search to inspect a cluster.")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="", help="EVOC cache dir")
    parser.add_argument("--results-key", default=None, help="specific results key")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    args = parser.parse_args()

    searcher = EvocClusterSearch(cache_dir=args.cache_dir or "", results_key=args.results_key)
    app = build_app(searcher)
    app.run_server(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
