"""
cluster_viz.py
==============
Standalone visualization module for the hierarchical clustering pipeline.
Drop this in the same directory as cluster_pipeline.py and call it after main().

Provides six visualization methods:

    1. parallel_coords()   — Parallel coordinates (like the reference image):
                             each line = one cluster, axes = PCA components.
                             Brush any axis to filter. Color = outer group.

    2. heatmap()           — Cluster × PCA-component heatmap. Shows which
                             components define each cluster's centroid.

    3. scatter_matrix()    — Pairwise scatter of top-4 PCA dims, colored by
                             outer group. Good for spotting linear separability.

    4. cluster_treemap()   — Treemap: outer groups as tiles, sub-clusters as
                             nested rectangles sized by member count.

    5. density_ridgeline() — Ridgeline (joy plot): one ridge per outer group,
                             distribution of a chosen PCA component. Reveals
                             multimodality inside groups.

    6. sunburst()          — Sunburst: outer group → sub-cluster hierarchy,
                             sized by member count. Good for proportion overview.

All methods return a self-contained HTML string AND optionally write it to disk.

Usage
-----
    from cluster_viz import ClusterViz

    viz = ClusterViz(
        embeddings      = embeddings,       # (N, D) float32 numpy
        group_labels    = group_labels,     # (N,)   int
        group_topics    = group_topics,     # list[str]
        sub_labels      = sub_labels,       # (N,)   int  (-1 = noise)
        combined_labels = combined_labels,  # (N,)   int
        combined_topics = combined_topics,  # dict[int, str]
        sentences       = sentences,        # list[str]
    )

    viz.parallel_coords(output_html="viz_parallel.html")
    viz.heatmap(output_html="viz_heatmap.html")
    viz.scatter_matrix(output_html="viz_scatter.html")
    viz.cluster_treemap(output_html="viz_treemap.html")
    viz.density_ridgeline(output_html="viz_ridgeline.html")
    viz.sunburst(output_html="viz_sunburst.html")

    # Or render all at once into a single tabbed dashboard:
    viz.dashboard(output_html="viz_dashboard.html")
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA


# ── Colour palette (matches pipeline) ────────────────────────────────────────
_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

_PAGE_CSS = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d0d14;
    color: #d4d4d8;
    font-family: 'JetBrains Mono', 'Fira Mono', monospace;
    min-height: 100vh;
  }
  #topbar {
    padding: 10px 20px;
    background: #13131f;
    border-bottom: 1px solid #2a2a40;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  #topbar h1 { font-size: 13px; color: #7c7cad; letter-spacing: .12em; text-transform: uppercase; }
  #topbar span { font-size: 11px; color: #444466; margin-left: auto; }
  #plot { width: 100%; height: calc(100vh - 48px); }
"""


def _html_shell(title: str, body_inner: str, extra_head: str = "") -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="{_PLOTLY_CDN}"></script>
<style>{_PAGE_CSS}</style>
{extra_head}
</head>
<body>
{body_inner}
</body>
</html>"""


def _write(html: str, path: Optional[str]) -> str:
    if path:
        Path(path).write_text(html, encoding="utf-8")
        print(f"  ✓ saved → {path}")
    return html


# =============================================================================
class ClusterViz:
    """
    Wraps all cluster-aware visualisations.
    Constructed once from pipeline outputs, then call individual methods.
    """

    def __init__(
        self,
        embeddings      : np.ndarray,
        group_labels    : np.ndarray,
        group_topics    : list[str],
        sub_labels      : np.ndarray,
        combined_labels : np.ndarray,
        combined_topics : dict[int, str],
        sentences       : list[str],
        n_pca_dims      : int = 8,
    ):
        self.embeddings      = embeddings
        self.group_labels    = group_labels
        self.group_topics    = group_topics
        self.sub_labels      = sub_labels
        self.combined_labels = combined_labels
        self.combined_topics = combined_topics
        self.sentences       = sentences
        self.N               = len(sentences)
        self.n_groups        = len(group_topics)

        # Reduce to n_pca_dims for all 2-D views
        actual_dims = min(n_pca_dims, embeddings.shape[1], self.N - 1)
        print(f"  Computing PCA({actual_dims}) for visualisations …")
        pca = PCA(n_components=actual_dims, random_state=42)
        self.pca_coords = pca.fit_transform(embeddings).astype(np.float32)
        self.pca_dims   = actual_dims
        self.pca_var    = pca.explained_variance_ratio_

        # Short topic labels
        self.short_topics = [t.split(",")[0].strip()[:30] for t in group_topics]

        # Per-cluster centroid in PCA space
        self._cluster_centroids: dict[int, np.ndarray] = {}
        for cl in sorted(set(combined_labels)):
            if cl == -1:
                continue
            mask = combined_labels == cl
            self._cluster_centroids[cl] = self.pca_coords[mask].mean(axis=0)

    # =========================================================================
    # 1. Parallel coordinates
    # =========================================================================

    def parallel_coords(
        self,
        output_html: Optional[str] = "viz_parallel.html",
        max_lines: int = 3000,
    ) -> str:
        """
        Parallel coordinates plot — one line per data point, axes = PCA dims.
        Lines are coloured by outer group.  Brush any axis to filter interactively.
        Large datasets are downsampled proportionally per group to max_lines total.
        """
        rng = np.random.default_rng(42)

        # Downsample if needed
        if self.N > max_lines:
            chosen = []
            for g in range(self.n_groups):
                idx = np.where(self.group_labels == g)[0]
                k   = max(1, round(max_lines * len(idx) / self.N))
                chosen.extend(rng.choice(idx, size=min(k, len(idx)), replace=False).tolist())
            chosen = np.array(chosen)
        else:
            chosen = np.arange(self.N)

        coords  = self.pca_coords[chosen]
        glabels = self.group_labels[chosen]

        # Build dimensions list
        dims = []
        for d in range(self.pca_dims):
            vals = coords[:, d].tolist()
            dims.append({
                "label"  : f"PC{d+1}",
                "values" : vals,
                "range"  : [float(np.min(vals)), float(np.max(vals))],
            })

        # Colour map: group index → colour index
        color_vals = glabels.tolist()

        # Hover: group topic
        customdata = [self.short_topics[g] for g in glabels]

        trace = {
            "type"       : "parcoords",
            "line"       : {
                "color"           : color_vals,
                "colorscale"      : [[i / max(self.n_groups - 1, 1), _PALETTE[i % len(_PALETTE)]]
                                     for i in range(self.n_groups)],
                "showscale"       : False,
                "opacity"         : 0.35,
            },
            "dimensions" : dims,
            "labelangle" : -20,
            "labelfont"  : {"size": 11, "color": "#9090c0"},
            "tickfont"   : {"size": 9,  "color": "#555580"},
            "rangefont"  : {"size": 9,  "color": "#555580"},
        }

        layout = {
            "paper_bgcolor" : "#0d0d14",
            "plot_bgcolor"  : "#0d0d14",
            "font"          : {"color": "#d4d4d8", "family": "monospace"},
            "margin"        : {"l": 60, "r": 60, "t": 40, "b": 40},
        }

        data_json   = json.dumps([trace])
        layout_json = json.dumps(layout)
        n_shown     = len(chosen)
        subtitle    = f"{n_shown} of {self.N} points · {self.n_groups} groups · {self.pca_dims} PCA axes"

        body = f"""
<div id="topbar">
  <h1>Parallel Coordinates — PCA Components by Cluster</h1>
  <span>{subtitle}</span>
</div>
<div id="plot"></div>
<script>
Plotly.newPlot('plot', {data_json}, {layout_json}, {{responsive:true}});
</script>"""

        html = _html_shell("Parallel Coordinates", body)
        return _write(html, output_html)

    # =========================================================================
    # 2. Cluster centroid heatmap
    # =========================================================================

    def heatmap(
        self,
        output_html: Optional[str] = "viz_heatmap.html",
    ) -> str:
        """
        Heatmap: rows = sub-clusters, columns = PCA components.
        Cell value = mean PCA coordinate of that cluster on that component.
        Reveals which components define each cluster.
        """
        cluster_ids = sorted(self._cluster_centroids.keys())
        if not cluster_ids:
            print("  ⚠ No clusters to plot.")
            return ""

        # Cap at 80 clusters for readability
        if len(cluster_ids) > 80:
            cluster_ids = cluster_ids[:80]

        matrix     = np.stack([self._cluster_centroids[c] for c in cluster_ids])
        # Z-score each column so all components are on same scale
        col_mean   = matrix.mean(axis=0)
        col_std    = matrix.std(axis=0) + 1e-8
        matrix_z   = (matrix - col_mean) / col_std

        row_labels = []
        for c in cluster_ids:
            topic = self.combined_topics.get(c, str(c))
            inner = topic.split("›")[-1].strip() if "›" in topic else topic
            row_labels.append(inner[:35])

        col_labels = [f"PC{i+1}" for i in range(self.pca_dims)]

        trace = {
            "type"        : "heatmap",
            "z"           : matrix_z.tolist(),
            "x"           : col_labels,
            "y"           : row_labels,
            "colorscale"  : "RdBu",
            "reversescale": True,
            "zmid"        : 0,
            "hoverongaps" : False,
            "hovertemplate": "Cluster: %{y}<br>%{x}: %{z:.3f}<extra></extra>",
            "colorbar"    : {
                "title"     : "Z-score",
                "titlefont" : {"size": 10, "color": "#9090c0"},
                "tickfont"  : {"size": 9,  "color": "#9090c0"},
                "thickness" : 12,
            },
        }

        layout = {
            "paper_bgcolor" : "#0d0d14",
            "plot_bgcolor"  : "#0d0d14",
            "font"          : {"color": "#d4d4d8", "family": "monospace", "size": 10},
            "margin"        : {"l": 220, "r": 80, "t": 40, "b": 60},
            "xaxis"         : {"side": "top", "tickangle": -30, "gridcolor": "#1e1e30"},
            "yaxis"         : {"autorange": "reversed", "tickfont": {"size": 9}},
        }

        data_json   = json.dumps([trace])
        layout_json = json.dumps(layout)
        n_shown     = len(cluster_ids)

        body = f"""
<div id="topbar">
  <h1>Cluster Centroid Heatmap — Z-scored PCA Components</h1>
  <span>{n_shown} sub-clusters · {self.pca_dims} components</span>
</div>
<div id="plot"></div>
<script>
Plotly.newPlot('plot', {data_json}, {layout_json}, {{responsive:true}});
</script>"""

        html = _html_shell("Cluster Heatmap", body)
        return _write(html, output_html)

    # =========================================================================
    # 3. Scatter matrix (SPLOM)
    # =========================================================================

    def scatter_matrix(
        self,
        output_html: Optional[str] = "viz_scatter.html",
        n_dims: int = 4,
        max_points: int = 5000,
    ) -> str:
        """
        Pairwise scatter matrix of the top n_dims PCA components.
        Each cell = 2-D scatter of two components. Color = outer group.
        Good for spotting linear separability between groups.
        """
        rng   = np.random.default_rng(42)
        n_dim = min(n_dims, self.pca_dims)
        idx   = (rng.choice(self.N, size=min(max_points, self.N), replace=False)
                 if self.N > max_points else np.arange(self.N))

        coords  = self.pca_coords[idx, :n_dim]
        glabels = self.group_labels[idx]

        traces = []
        for g in range(self.n_groups):
            mask = glabels == g
            if not mask.any():
                continue
            gcoords = coords[mask]
            traces.append({
                "type"       : "splom",
                "name"       : self.short_topics[g],
                "dimensions" : [
                    {"label": f"PC{d+1}", "values": gcoords[:, d].tolist()}
                    for d in range(n_dim)
                ],
                "marker"     : {
                    "color"  : _PALETTE[g % len(_PALETTE)],
                    "size"   : 3,
                    "opacity": 0.5,
                    "line"   : {"width": 0},
                },
                "showupperhalf": False,
                "diagonal"   : {"visible": True},
            })

        layout = {
            "paper_bgcolor" : "#0d0d14",
            "plot_bgcolor"  : "#0d0d14",
            "font"          : {"color": "#d4d4d8", "family": "monospace", "size": 9},
            "legend"        : {"font": {"size": 9}, "itemsizing": "constant"},
            "margin"        : {"l": 40, "r": 20, "t": 40, "b": 40},
        }

        data_json   = json.dumps(traces)
        layout_json = json.dumps(layout)
        n_shown     = len(idx)

        body = f"""
<div id="topbar">
  <h1>Scatter Matrix — Top {n_dim} PCA Components</h1>
  <span>{n_shown} points · {self.n_groups} groups</span>
</div>
<div id="plot"></div>
<script>
Plotly.newPlot('plot', {data_json}, {layout_json}, {{responsive:true}});
</script>"""

        html = _html_shell("Scatter Matrix", body)
        return _write(html, output_html)

    # =========================================================================
    # 4. Cluster treemap
    # =========================================================================

    def cluster_treemap(
        self,
        output_html: Optional[str] = "viz_treemap.html",
    ) -> str:
        """
        Treemap: outer groups as large tiles, sub-clusters as nested rectangles.
        Rectangle area = member count.  Hover shows topic and size.
        """
        ids     = ["root"]
        labels  = ["All clusters"]
        parents = [""]
        values  = [0]
        colors  = ["#0d0d14"]
        texts   = [""]

        for g in range(self.n_groups):
            g_id    = f"g_{g}"
            g_count = int((self.group_labels == g).sum())
            ids.append(g_id)
            labels.append(self.short_topics[g])
            parents.append("root")
            values.append(g_count)
            colors.append(_PALETTE[g % len(_PALETTE)])
            texts.append(f"{g_count} titles")

            # Sub-clusters inside this group
            g_mask    = self.group_labels == g
            g_combined = self.combined_labels[g_mask]
            for sub_id in sorted(set(g_combined) - {-1}):
                cl      = g * 10_000 + sub_id
                cl_id   = f"cl_{cl}"
                cl_mask = self.combined_labels == cl
                count   = int(cl_mask.sum())
                topic   = self.combined_topics.get(cl, f"sub {sub_id}")
                inner   = topic.split("›")[-1].strip() if "›" in topic else topic
                ids.append(cl_id)
                labels.append(inner[:30])
                parents.append(g_id)
                values.append(count)
                colors.append(_PALETTE[g % len(_PALETTE)] + "bb")
                texts.append(f"{count} titles")

        trace = {
            "type"         : "treemap",
            "ids"          : ids,
            "labels"       : labels,
            "parents"      : parents,
            "values"       : values,
            "text"         : texts,
            "hovertemplate": "<b>%{label}</b><br>%{text}<extra></extra>",
            "marker"       : {
                "colors"     : colors,
                "line"       : {"width": 1, "color": "#0d0d14"},
            },
            "textfont"     : {"size": 10, "color": "#ffffff"},
            "pathbar"      : {"visible": True, "thickness": 20,
                              "textfont": {"size": 10, "color": "#d4d4d8"}},
            "branchvalues" : "total",
        }

        layout = {
            "paper_bgcolor" : "#0d0d14",
            "font"          : {"color": "#d4d4d8", "family": "monospace"},
            "margin"        : {"l": 10, "r": 10, "t": 40, "b": 10},
        }

        data_json   = json.dumps([trace])
        layout_json = json.dumps(layout)

        body = f"""
<div id="topbar">
  <h1>Cluster Treemap — Outer Groups → Sub-clusters</h1>
  <span>{self.n_groups} groups · {len(self._cluster_centroids)} sub-clusters · Click to drill in</span>
</div>
<div id="plot"></div>
<script>
Plotly.newPlot('plot', {data_json}, {layout_json}, {{responsive:true}});
</script>"""

        html = _html_shell("Cluster Treemap", body)
        return _write(html, output_html)

    # =========================================================================
    # 5. Density ridgeline (joy plot)
    # =========================================================================

    def density_ridgeline(
        self,
        output_html : Optional[str] = "viz_ridgeline.html",
        component   : int = 0,
        max_groups  : int = 20,
    ) -> str:
        """
        Ridgeline / joy plot: one ridge per outer group, showing the distribution
        of a chosen PCA component.  Reveals multimodality within groups.
        component: 0-indexed PCA dimension to plot on the x-axis.
        """
        comp = min(component, self.pca_dims - 1)
        vals = self.pca_coords[:, comp]

        n_groups_show = min(self.n_groups, max_groups)
        # Sort groups by median value on this component
        medians = [
            (g, float(np.median(vals[self.group_labels == g])))
            for g in range(n_groups_show)
        ]
        medians.sort(key=lambda x: x[1])

        traces  = []
        spacing = 1.0   # y-offset per group

        for rank, (g, _med) in enumerate(medians):
            gvals = vals[self.group_labels == g].tolist()
            color = _PALETTE[g % len(_PALETTE)]
            y_off = rank * spacing

            traces.append({
                "type"       : "violin",
                "name"       : self.short_topics[g],
                "x"          : gvals,
                "y0"         : y_off,
                "orientation": "h",
                "side"       : "positive",
                "line"       : {"color": color, "width": 1.5},
                "fillcolor"  : color + "55",
                "meanline"   : {"visible": True, "color": color, "width": 1},
                "points"     : False,
                "spanmode"   : "soft",
                "showlegend" : True,
                "hoveron"    : "violins",
                "hovertemplate": f"<b>{self.short_topics[g]}</b><br>PC{comp+1}: %{{x:.3f}}<extra></extra>",
            })

        tick_vals = [rank * spacing for rank, _ in enumerate(medians)]
        tick_text = [self.short_topics[g] for _, (g, _) in enumerate(medians)]

        layout = {
            "paper_bgcolor" : "#0d0d14",
            "plot_bgcolor"  : "#0d0d14",
            "font"          : {"color": "#d4d4d8", "family": "monospace", "size": 10},
            "violinmode"    : "overlay",
            "violingap"     : 0,
            "violingroupgap": 0,
            "showlegend"    : False,
            "margin"        : {"l": 180, "r": 40, "t": 40, "b": 50},
            "xaxis"         : {
                "title"    : f"PC{comp+1}  (explains {100*self.pca_var[comp]:.1f}% variance)",
                "gridcolor": "#1e1e30", "zeroline": False,
            },
            "yaxis"         : {
                "tickvals"  : tick_vals,
                "ticktext"  : tick_text,
                "tickfont"  : {"size": 9},
                "gridcolor" : "#1e1e30",
                "zeroline"  : False,
                "showgrid"  : False,
            },
        }

        data_json   = json.dumps(traces)
        layout_json = json.dumps(layout)

        body = f"""
<div id="topbar">
  <h1>Density Ridgeline — PC{comp+1} Distribution per Group</h1>
  <span>{n_groups_show} groups · PC{comp+1} explains {100*self.pca_var[comp]:.1f}% variance</span>
</div>
<div id="plot"></div>
<script>
Plotly.newPlot('plot', {data_json}, {layout_json}, {{responsive:true}});
</script>"""

        html = _html_shell("Density Ridgeline", body)
        return _write(html, output_html)

    # =========================================================================
    # 6. Sunburst
    # =========================================================================

    def sunburst(
        self,
        output_html: Optional[str] = "viz_sunburst.html",
    ) -> str:
        """
        Sunburst chart: inner ring = outer groups, outer ring = sub-clusters.
        Sized by member count.  Click to zoom into a group.
        """
        ids     = ["root"]
        labels  = ["Clusters"]
        parents = [""]
        values  = [0]
        colors  = ["#0d0d14"]

        for g in range(self.n_groups):
            g_id    = f"g_{g}"
            g_count = int((self.group_labels == g).sum())
            ids.append(g_id)
            labels.append(self.short_topics[g])
            parents.append("root")
            values.append(g_count)
            colors.append(_PALETTE[g % len(_PALETTE)])

            g_mask     = self.group_labels == g
            g_combined = self.combined_labels[g_mask]
            for sub_id in sorted(set(g_combined) - {-1}):
                cl      = g * 10_000 + sub_id
                cl_mask = self.combined_labels == cl
                count   = int(cl_mask.sum())
                topic   = self.combined_topics.get(cl, f"sub {sub_id}")
                inner   = topic.split("›")[-1].strip() if "›" in topic else topic
                ids.append(f"cl_{cl}")
                labels.append(inner[:25])
                parents.append(g_id)
                values.append(count)
                colors.append(_PALETTE[g % len(_PALETTE)] + "99")

        trace = {
            "type"         : "sunburst",
            "ids"          : ids,
            "labels"       : labels,
            "parents"      : parents,
            "values"       : values,
            "branchvalues" : "total",
            "marker"       : {"colors": colors, "line": {"width": 1, "color": "#0d0d14"}},
            "leaf"         : {"opacity": 0.85},
            "hovertemplate": "<b>%{label}</b><br>%{value} titles<extra></extra>",
            "textfont"     : {"size": 10},
            "insidetextorientation": "radial",
            "maxdepth"     : 2,
        }

        layout = {
            "paper_bgcolor" : "#0d0d14",
            "font"          : {"color": "#d4d4d8", "family": "monospace"},
            "margin"        : {"l": 10, "r": 10, "t": 40, "b": 10},
        }

        data_json   = json.dumps([trace])
        layout_json = json.dumps(layout)

        body = f"""
<div id="topbar">
  <h1>Cluster Sunburst — Group → Sub-cluster Hierarchy</h1>
  <span>{self.n_groups} groups · {len(self._cluster_centroids)} sub-clusters · Click to zoom</span>
</div>
<div id="plot"></div>
<script>
Plotly.newPlot('plot', {data_json}, {layout_json}, {{responsive:true}});
</script>"""

        html = _html_shell("Cluster Sunburst", body)
        return _write(html, output_html)

    # =========================================================================
    # 7. Tabbed dashboard (all views in one file)
    # =========================================================================

    def dashboard(
        self,
        output_html: Optional[str] = "viz_dashboard.html",
        max_lines  : int = 3000,
    ) -> str:
        """
        Single-file tabbed dashboard containing all six visualisations.
        No network requests beyond Plotly CDN.
        """
        print("  Building dashboard (all 6 views) …")

        # Build each view without writing to disk (pass output_html=None)
        views = {
            "Parallel Coords" : self.parallel_coords(output_html=None, max_lines=max_lines),
            "Heatmap"         : self.heatmap(output_html=None),
            "Scatter Matrix"  : self.scatter_matrix(output_html=None),
            "Treemap"         : self.cluster_treemap(output_html=None),
            "Ridgeline"       : self.density_ridgeline(output_html=None),
            "Sunburst"        : self.sunburst(output_html=None),
        }

        # Extract just the <body> inner HTML + inline <script> from each view
        # (strip the outer shell — we'll wrap in our own tabbed shell)
        def _extract_body(full_html: str) -> str:
            """Pull content between <body> and </body>, minus the topbar."""
            start = full_html.find("<body>") + 6
            end   = full_html.rfind("</body>")
            inner = full_html[start:end].strip()
            # Remove the topbar div (we have our own tab bar)
            tb_s = inner.find('<div id="topbar">')
            tb_e = inner.find("</div>", tb_s) + 6
            inner = inner[tb_e:].strip()
            return inner

        tabs_html  = ""
        panels_html = ""

        for i, (name, view_html) in enumerate(views.items()):
            active     = "active" if i == 0 else ""
            tab_id     = f"tab_{i}"
            panel_id   = f"panel_{i}"
            body_inner = _extract_body(view_html)
            tabs_html  += f'<button class="tab {active}" onclick="switchTab({i})" id="{tab_id}">{name}</button>\n'
            panels_html += f"""
<div class="panel {'visible' if i == 0 else 'hidden'}" id="{panel_id}">
  <div id="plot_{i}" style="width:100%;height:calc(100vh - 92px)"></div>
  {body_inner.replace('id="plot"', f'id="plot_{i}_inner"').replace("'plot'", f"'plot_{i}'")}
</div>"""

        tab_css = """
  #tabbar {
    display: flex;
    gap: 2px;
    padding: 6px 16px;
    background: #0a0a12;
    border-bottom: 1px solid #1e1e30;
    flex-shrink: 0;
  }
  .tab {
    padding: 5px 16px;
    background: #13131f;
    border: 1px solid #2a2a40;
    border-radius: 3px 3px 0 0;
    color: #6060a0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    letter-spacing: .06em;
    transition: background .15s, color .15s;
  }
  .tab:hover  { background: #1e1e35; color: #a0a0d0; }
  .tab.active { background: #1e1e35; color: #d4d4f8; border-bottom-color: #1e1e35; }
  .panel.hidden  { display: none; }
  .panel.visible { display: block; }
"""

        tab_js = """
const PANELS = document.querySelectorAll('.panel');
const TABS   = document.querySelectorAll('.tab');
const rendered = new Set([0]);

function switchTab(i) {
  TABS.forEach((t, idx) => t.classList.toggle('active', idx === i));
  PANELS.forEach((p, idx) => {
    p.classList.toggle('visible', idx === i);
    p.classList.toggle('hidden',  idx !== i);
  });
  // Trigger Plotly resize so it fills the newly visible panel
  setTimeout(() => {
    const plots = document.querySelectorAll(`#panel_${i} [id^="plot_"]`);
    plots.forEach(el => { try { Plotly.relayout(el.id, {}); } catch(e){} });
  }, 50);
}
"""

        full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Cluster Dashboard</title>
<script src="{_PLOTLY_CDN}"></script>
<style>
{_PAGE_CSS}
{tab_css}
body {{ display: flex; flex-direction: column; }}
</style>
</head>
<body>
<div id="topbar" style="flex-shrink:0">
  <h1>Cluster Dashboard</h1>
  <span>{self.n_groups} groups · {len(self._cluster_centroids)} sub-clusters · {self.N:,} titles</span>
</div>
<div id="tabbar">
{tabs_html}
</div>
{panels_html}
<script>
{tab_js}
</script>
</body>
</html>"""

        return _write(full_html, output_html)


# =============================================================================
# Convenience function — call after main() in cluster_pipeline.py
# =============================================================================

def build_all_visualisations(
    embeddings      : np.ndarray,
    group_labels    : np.ndarray,
    group_topics    : list[str],
    sub_labels      : np.ndarray,
    combined_labels : np.ndarray,
    combined_topics : dict[int, str],
    sentences       : list[str],
    out_dir         : str = ".",
    dashboard_only  : bool = False,
) -> ClusterViz:
    """
    Convenience wrapper: creates ClusterViz and renders all HTMLs.

    Parameters
    ----------
    dashboard_only : bool
        If True, only generate the single tabbed dashboard file
        instead of six individual files + the dashboard.

    Returns the ClusterViz instance so you can call individual methods later.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    viz = ClusterViz(
        embeddings      = embeddings,
        group_labels    = group_labels,
        group_topics    = group_topics,
        sub_labels      = sub_labels,
        combined_labels = combined_labels,
        combined_topics = combined_topics,
        sentences       = sentences,
    )

    if dashboard_only:
        viz.dashboard(output_html=str(out / "viz_dashboard.html"))
    else:
        viz.parallel_coords (output_html=str(out / "viz_parallel.html"))
        viz.heatmap         (output_html=str(out / "viz_heatmap.html"))
        viz.scatter_matrix  (output_html=str(out / "viz_scatter.html"))
        viz.cluster_treemap (output_html=str(out / "viz_treemap.html"))
        viz.density_ridgeline(output_html=str(out / "viz_ridgeline.html"))
        viz.sunburst        (output_html=str(out / "viz_sunburst.html"))
        viz.dashboard       (output_html=str(out / "viz_dashboard.html"))

    return viz
