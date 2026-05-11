#!/usr/bin/env python3
"""Dash 3D drilldown visualization for EVoC hierarchies.

Features
- 3D scatter of cluster centroids for current view level
- Click a centroid to drill into its children (next finer level)
- Breadcrumb + back navigation
- Side panel with cluster metadata and sample docs

Run:
    python evoc_dash_3d.py --cache-dir /path/to/clustering_cache_evoc
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

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


def compute_3d(embs: np.ndarray) -> np.ndarray:
    """Try UMAP 3D, fallback to PCA 3D."""
    try:
        import umap

        reducer = umap.UMAP(n_components=3, random_state=42)
        return reducer.fit_transform(embs)
    except Exception:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3)
        return pca.fit_transform(embs)


def build_centroid_positions(searcher: EvocClusterSearch, proj: np.ndarray) -> list[dict[int, tuple[float, float, float]]]:
    """Return list indexed by level: mapping label -> centroid (x,y,z)."""
    out: list[dict[int, tuple[float, float, float]]] = []
    for level, label_to_indices in enumerate(searcher.layer_label_to_indices):
        cdict: dict[int, tuple[float, float, float]] = {}
        for label, ids in label_to_indices.items():
            if not ids:
                continue
            pts = proj[np.array(ids)]
            mean = pts.mean(axis=0)
            cdict[int(label)] = (float(mean[0]), float(mean[1]), float(mean[2]))
        out.append(cdict)
    return out


def build_app(searcher: EvocClusterSearch) -> Dash:
    app = Dash(__name__)

    embs = searcher.embeddings.astype(np.float32)
    proj = compute_3d(embs)

    layer_count = searcher.layer_count
    # searcher.search_layers: coarsest -> finest
    layers = searcher.search_layers

    # compute centroid positions in projected space for each level
    centroid_pos = build_centroid_positions(searcher, proj)

    # initial state: view level 0 (coarsest), parent_label = -1
    app.layout = html.Div([
        html.H3("EVoC 3D Drilldown"),
        html.Div(id="breadcrumb", style={"marginBottom": "6px"}),
        html.Div([
            html.Button("Back", id="back-btn"),
            html.Span(" "),
            html.Label("Level:"),
            dcc.Slider(id="level-slider", min=0, max=layer_count - 1, value=0, marks={i: str(i) for i in range(layer_count)}, step=1),
        ], style={"marginBottom": "8px"}),

        html.Div([
            dcc.Graph(id="graph-3d", style={"height": "700px", "width": "70%", "display": "inline-block"}),
            html.Div(id="side-panel", style={"width": "28%", "display": "inline-block", "verticalAlign": "top", "paddingLeft": "12px"}),
        ]),

        dcc.Store(id="state-store", data={"path": [], "view_level": 0}),
        dcc.Store(id="centroid-pos-store", data={"centroid_pos": centroid_pos}),
    ], style={"padding": "12px"})


    @app.callback(
        Output("breadcrumb", "children"),
        Input("state-store", "data"),
    )
    def render_breadcrumb(state):
        path = state.get("path", [])
        if not path:
            return html.Div([html.B("Root (level 0)")])
        crumbs = []
        for lvl, lbl in path:
            crumbs.append(html.Span(f"/ L{lvl}:{lbl} "))
        return html.Div([html.B("Path:"), *crumbs])


    @app.callback(
        Output("graph-3d", "figure"),
        Input("state-store", "data"),
        Input("level-slider", "value"),
        State("centroid-pos-store", "data"),
    )
    def update_graph(state, slider_level, centroid_store):
        view_level = state.get("view_level", 0)
        path = state.get("path", [])
        # slider overrides view level unless there's a path (drill mode)
        level = view_level if path else slider_level

        centroid_pos_local = centroid_store["centroid_pos"]
        pos_map = centroid_pos_local[level]

        labels = list(pos_map.keys())
        xs = [pos_map[l][0] for l in labels]
        ys = [pos_map[l][1] for l in labels]
        zs = [pos_map[l][2] for l in labels]
        sizes = [len(searcher.layer_label_to_indices[level].get(l, [])) for l in labels]
        topics = [searcher.layer_topics[level].get(l, "") for l in labels]

        df = pd.DataFrame({"label": labels, "x": xs, "y": ys, "z": zs, "size": sizes, "topic": topics})

        fig = px.scatter_3d(df, x="x", y="y", z="z", size="size", color=df["label"].astype(str), hover_name=df["label"].astype(str), hover_data=["size", "topic"]) 
        fig.update_traces(marker=dict(opacity=0.9, line=dict(width=0)), selector=dict(mode="markers"))
        fig.update_layout(scene=dict(xaxis_title=None, yaxis_title=None, zaxis_title=None))
        # attach customdata as [level,label]
        fig.data[0].customdata = np.stack([df["label"].values.astype(int), np.full(len(df), level)], axis=1).tolist()
        return fig


    @app.callback(
        Output("state-store", "data"),
        Output("side-panel", "children"),
        Input("graph-3d", "clickData"),
        Input("back-btn", "n_clicks"),
        State("state-store", "data"),
    )
    def drill_or_back(clickData, back_clicks, state):
        ctx = dash.callback_context
        triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else None
        path = state.get("path", [])
        view_level = state.get("view_level", 0)

        # Back button
        if triggered and triggered.startswith("back-btn"):
            if path:
                path = path[:-1]
                # update view_level to last path level or 0
                new_view = path[-1][0] + 1 if path else 0
                state = {"path": path, "view_level": new_view}
                return state, html.Div("Went back")
            return state, html.Div("At root")

        # Click centroid
        if clickData and "points" in clickData:
            pt = clickData["points"][0]
            cd = pt.get("customdata")
            if not cd:
                return state, html.Div("No selection data")
            label = int(cd[0])
            level = int(cd[1])

            # determine if label has children
            has_children = False
            if level < (searcher.layer_count - 1):
                children = searcher.parent_children[level].get(label, [])
                if children:
                    has_children = True

            if has_children:
                # drill into children: append (level,label) to path and set view_level to level+1
                path = path + [(level, label)]
                state = {"path": path, "view_level": level + 1}
                # show side panel for parent cluster
                indices = searcher.layer_label_to_indices[level].get(label, [])
                sample = [searcher.sentences[i] for i in indices[:50]]
                panel = html.Div([
                    html.H4(f"Drilled into L{level}:{label} — children: {len(children)}"),
                    html.Div([html.B("Topic: "), html.Span(searcher.layer_topics[level].get(label, ""))]),
                    html.H5("Sample docs in parent cluster"),
                    html.Div([html.Pre(s) for s in sample]),
                ])
                return state, panel
            else:
                # leaf cluster: show docs
                indices = searcher.layer_label_to_indices[level].get(label, [])
                sample = [searcher.sentences[i] for i in indices[:200]]
                panel = html.Div([
                    html.H4(f"Leaf L{level}:{label} — size {len(indices)}"),
                    html.Div([html.B("Topic: "), html.Span(searcher.layer_topics[level].get(label, ""))]),
                    html.H5("Documents"),
                    html.Div([html.Pre(s) for s in sample]),
                ])
                return state, panel

        return state, html.Div("Click a centroid to drill down.")

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
    print(f"Starting Dash 3D app on http://{args.host}:{args.port}")
    app.run_server(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
