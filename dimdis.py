import csv
import html
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch
import umap
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


_RE_SOURCE_SUFFIX = re.compile(r"\s+-\s+\S+\.\S+$")   # " - www.site.com" at end of title
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_NONALPHA = re.compile(r"[^a-zA-Z0-9\s\'\-]")
_RE_SPACES = re.compile(r"\s+")
_RE_SHORT_NUM = re.compile(r"(?<![a-zA-Z0-9\-])\d{1,3}(?![a-zA-Z0-9\-])")
_RE_THOUSANDS = re.compile(r"(\d),(\d)")

DEFAULT_CSV_LOCATIONS = [
    Path("train.csv"),
    Path("Easy Import") / "train.csv",
    Path("EVOC") / "train.csv",
    Path("archive") / "train.csv",
]


def clean_title(raw: str) -> str:
    """
    Normalise a news headline for both TF-IDF and embedding.

    Steps (in order):
      1. html.unescape x2 : handles double-encoded entities (common in RSS)
         &#39; -> '   &amp; -> &   &lt; -> <   &gt; -> >
         &amp;#39; -> &#39; -> ' (double-encoded)
      2. Strip source suffix: "Some Title - www.washingtonpost.com" -> "Some Title"
      3. Strip residual HTML tags (<b>, <i>, etc.)
      4. Collapse formatted numbers: "1 000 jobs" -> "1000 jobs"
         Prevents "000" token surviving after comma-stripping.
      5. Remove punctuation that isn't apostrophe or hyphen
         (keeps contractions and hyphenated words intact)
      6. Remove bare short-digit tokens (1-3 digits) left by HTML entity
         residue, e.g. the "39" in "don 39t" that comes from &#39; -> ' -> 39
         after apostrophe stripping. Real years (4 digits) are kept.
      7. Collapse whitespace and strip
    """
    s = html.unescape(raw)                  # first pass: &amp;#39; -> &#39;
    s = html.unescape(s)                    # second pass: &#39; -> '
    s = _RE_SOURCE_SUFFIX.sub("", s)        # "Title - site.com" -> "Title"
    s = _RE_HTML_TAG.sub(" ", s)            # <b>foo</b> -> foo
    s = _RE_THOUSANDS.sub(r"\1\2", s)      # "1,000" -> "1000" before comma-strip
    s = _RE_THOUSANDS.sub(r"\1\2", s)      # second pass for "1,000,000"
    s = _RE_NONALPHA.sub(" ", s)            # remove remaining punctuation / symbols
    s = _RE_SHORT_NUM.sub(" ", s)           # remove bare 1-3 digit tokens (entity residue)
    s = _RE_SPACES.sub(" ", s)              # collapse whitespace
    return s.strip()


def resolve_csv_path(csv_path: Path | None) -> Path:
    if csv_path is not None and csv_path.exists():
        return csv_path
    for candidate in DEFAULT_CSV_LOCATIONS:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(p) for p in DEFAULT_CSV_LOCATIONS)
    raise FileNotFoundError(
        "Could not find train.csv. Tried: " + tried +
        ". Provide --csv PATH to specify it explicitly."
    )


def load_titles_from_csv(csv_file: Path) -> list[str]:
    """Load and clean non-empty values from the 'Title' column of a CSV file."""
    titles: list[str] = []
    with csv_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("Title") or "").strip()
            if not raw:
                continue
            cleaned = clean_title(raw)
            if cleaned:  # skip titles that become empty after cleaning
                titles.append(cleaned)
    return titles


def embed_titles(
    titles: Iterable[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        list(titles),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings


def reduce_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


# === Helpers: clustering / diagnostics / printing ===
def print_repr_info(name: str, X: np.ndarray, max_rows: int = 3):
    """Print intuitive summary about a representation X (n_samples x dims)."""
    n, d = X.shape
    norms = np.linalg.norm(X, axis=1)
    print(f"\nRepresentation: {name}")
    print(f" - shape: {n} samples x {d} dims")
    print(f" - dtype: {X.dtype}")
    print(f" - norms: mean={norms.mean():.3f}, std={norms.std():.3f}, min={norms.min():.3f}, max={norms.max():.3f}")
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    print(f" - per-dim range (first 3): [{mins[:3].round(3)}] to [{maxs[:3].round(3)}]")
    print(f" - sample rows (first {min(max_rows,n)}):")
    for i in range(min(max_rows, n)):
        print(f"    [{i}] {np.array2string(X[i][:6], precision=3, separator=', ')} ...")


def hopkins_statistic(X: np.ndarray, m: int | None = None, random_state: int | None = None) -> float:
    """Compute the Hopkins statistic for cluster tendency.

    H ~ 1 indicates highly clusterable, ~0.5 random.
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    if m is None:
        m = min(50, n - 1)

    # Use cuML NearestNeighbors if available
    if 'CUML_AVAILABLE' in globals() and CUML_AVAILABLE:
        try:
            cp = CUPY
            cuNN = CUML['NearestNeighbors']
            X_gpu = cp.asarray(X)
            nn = cuNN(n_neighbors=2)
            nn.fit(X_gpu)

            idx = rng.choice(n, size=m, replace=False)
            idx_gpu = cp.asarray(idx)
            distances_w_gpu, _ = nn.kneighbors(X_gpu[idx_gpu], n_neighbors=2)
            w = cp.asnumpy(distances_w_gpu)[:, 1]

            mins = X.min(axis=0)
            maxs = X.max(axis=0)
            u_points = rng.uniform(mins, maxs, size=(m, d))
            u_gpu = cp.asarray(u_points)
            distances_u_gpu, _ = nn.kneighbors(u_gpu, n_neighbors=1)
            u = cp.asnumpy(distances_u_gpu)[:, 0]

            H = u.sum() / (u.sum() + w.sum())
            return float(H)
        except Exception:
            pass

    # Fallback: sklearn NearestNeighbors on CPU
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(n, size=m, replace=False)
    distances_w, _ = nbrs.kneighbors(X[idx], n_neighbors=2)
    w = distances_w[:, 1]
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    u_points = rng.uniform(mins, maxs, size=(m, d))
    distances_u, _ = nbrs.kneighbors(u_points, n_neighbors=1)
    u = distances_u[:, 0]
    H = u.sum() / (u.sum() + w.sum())
    return float(H)


def twonn_intrinsic_dim(X: np.ndarray) -> float:
    """Estimate intrinsic dimension using the TwoNN method (Facco et al.)."""
    n = X.shape[0]
    if n < 3:
        return float('nan')

    # Prefer cuML NearestNeighbors on GPU when available
    if 'CUML_AVAILABLE' in globals() and CUML_AVAILABLE:
        try:
            cp = CUPY
            cuNN = CUML['NearestNeighbors']
            X_gpu = cp.asarray(X)
            nn = cuNN(n_neighbors=3)
            nn.fit(X_gpu)
            dists_gpu, _ = nn.kneighbors(X_gpu)
            dists = cp.asnumpy(dists_gpu)
        except Exception:
            dists = None
    else:
        dists = None

    if dists is None:
        nbrs = NearestNeighbors(n_neighbors=3).fit(X)
        dists, _ = nbrs.kneighbors(X)

    r1 = dists[:, 1]
    r2 = dists[:, 2]
    # avoid zero distances
    mask = (r1 > 0) & (r2 > 0)
    r1 = r1[mask]
    r2 = r2[mask]
    mu = r2 / r1
    mu_sorted = np.sort(mu)
    # empirical CDF values for sorted mu
    cdf = np.arange(1, len(mu_sorted) + 1) / len(mu_sorted)
    # avoid 1.0 in log(1 - cdf)
    eps = 1e-10
    y = np.log(1.0 - cdf + eps)
    x = np.log(mu_sorted + eps)
    # linear fit y = a + b*x -> slope b, intrinsic dim d = -b
    b, a = np.polyfit(x, y, 1)
    d = -b
    return float(d)


def ripser_cluster_summary(X: np.ndarray, thresh: float = 0.1) -> dict:
    """Use ripser to compute simple persistence-based cluster summary.

    Returns counts of persistent H0/H1 features above a relative threshold.
    """
    try:
        from ripser import ripser
    except Exception:
        return {"ripser_available": False}
    res = ripser(X, maxdim=1)
    dgms = res.get('dgms', [])
    summary = {"ripser_available": True}
    if len(dgms) > 0:
        # H0 diagram: dgms[0] contains (birth, death) pairs; death may be inf
        h0 = np.array(dgms[0])
        # persistence = death - birth; treat inf death as large number
        deaths = h0[:, 1]
        births = h0[:, 0]
        finite_deaths = np.where(np.isfinite(deaths), deaths, deaths[~np.isinf(deaths)].max() if np.any(~np.isinf(deaths)) else 1.0)
        persistence = finite_deaths - births
        max_p = persistence.max() if persistence.size > 0 else 0.0
        # count components with persistence > thresh * max_p
        if max_p > 0:
            count_h0 = int((persistence > (thresh * max_p)).sum())
        else:
            count_h0 = 0
        summary.update({"H0_persistent_count": count_h0, "H0_max_persistence": float(max_p)})
    if len(dgms) > 1:
        h1 = np.array(dgms[1])
        if h1.size:
            p1 = (h1[:, 1] - h1[:, 0])
            summary.update({"H1_max_persistence": float(p1.max())})
    return summary


def compute_best_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int | None = 42):
    """Run KMeans for k in [k_min..k_max] and return best (k, score, labels).

    Returns (best_k, best_score, best_labels) or (None, nan, None) if not computable.
    """
    def fit_kmeans_labels(X_local: np.ndarray, n_clusters: int, random_state_local: int | None = 42):
        # GPU-backed KMeans via cuML when available, else sklearn KMeans
        if 'CUML_AVAILABLE' in globals() and CUML_AVAILABLE:
            try:
                cp = CUPY
                cuKMeans = CUML['KMeans']
                X_gpu = cp.asarray(X_local)
                km = cuKMeans(n_clusters=n_clusters, random_state=random_state_local)
                labels_gpu = km.fit_predict(X_gpu)
                return cp.asnumpy(labels_gpu)
            except Exception:
                pass
        # fallback CPU
        km = KMeans(n_clusters=n_clusters, random_state=random_state_local, n_init=10)
        return km.fit_predict(X_local)

    n = X.shape[0]
    if n < 2:
        return None, float('nan'), None
    k_max = min(k_max, n - 1)
    best_k = None
    best_score = float('-inf')
    best_labels = None
    for k in range(k_min, max(k_min, k_max) + 1):
        if k >= n:
            break
        try:
            labels = fit_kmeans_labels(X, k, random_state)
            if labels is None or len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception:
            continue
    if best_k is None:
        return None, float('nan'), None
    return best_k, float(best_score), best_labels


def run_clusterability_checks(name: str, points: np.ndarray):
    print(f"\nClusterability checks for {name} (n={points.shape[0]}):")
    hop = hopkins_statistic(points)
    twonn = twonn_intrinsic_dim(points)
    rip = ripser_cluster_summary(points)
    print(f"- Hopkins statistic: {hop:.4f}  (close to 1 => clusterable)")
    print(f"- TwoNN intrinsic dimension estimate: {twonn:.3f}")
    if rip.get("ripser_available"):
        print(f"- Ripser H0 persistent count: {rip.get('H0_persistent_count', 0)}")
        print(f"- Ripser H0 max persistence: {rip.get('H0_max_persistence', 0.0):.4f}")
        if 'H1_max_persistence' in rip:
            print(f"- Ripser H1 max persistence: {rip.get('H1_max_persistence'):.4f}")
    else:
        print("- Ripser not installed; skipping persistence checks.")



def make_plot(points: np.ndarray, titles: list[str], title: str):
    df = {
        "x": points[:, 0],
        "y": points[:, 1],
        "z": points[:, 2],
        "title": titles,
    }
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        hover_name="title",
        title=title,
        height=720,
    )
    fig.update_traces(marker=dict(size=4, opacity=0.8))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available. Install a CUDA-enabled PyTorch "
            "build or use --device cpu."
        )
    return requested


def try_reduce_with_cuml(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
):
    try:
        import cupy as cp
        from cuml import PCA as cuPCA
        from cuml import UMAP as cuUMAP
    except Exception:
        return None

    gpu_embeddings = cp.asarray(embeddings)
    pca = cuPCA(n_components=n_components, random_state=42)
    umap_reducer = cuUMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    pca_points = pca.fit_transform(gpu_embeddings)
    umap_points = umap_reducer.fit_transform(gpu_embeddings)
    return cp.asnumpy(pca_points), cp.asnumpy(umap_points)


# Configuration (was previously CLI args). Edit these variables directly.
# Set `CSV_PATH_OVERRIDE` to a Path to point to a specific CSV, or leave None
# to search default locations.
CSV_PATH_OVERRIDE: Path | None = None
MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 64
UMAP_NEIGHBORS = 5
UMAP_MIN_DIST = 0.05
OUTPUT_DIR = Path("outputs")
DEVICE_REQUEST = "auto"  # one of: "auto", "cpu", "cuda"
USE_CUML = False
NO_SHOW = False

# Resolve runtime values from the configuration above
csv_path = resolve_csv_path(CSV_PATH_OVERRIDE)
device = select_device(DEVICE_REQUEST)

titles = load_titles_from_csv(csv_path)
if not titles:
    raise ValueError("No valid titles found in the CSV file.")

embeddings = embed_titles(titles, MODEL, BATCH_SIZE, device)

# Auto-detect and enable cuML/cuPy when running on CUDA and available
CUPY = None
CUML = None
CUML_AVAILABLE = False
if device == "cuda":
    try:
        import cupy as cp
        # attempt to import a few cuML classes used below
        from cuml.cluster import KMeans as _cuKMeans  # type: ignore
        from cuml.neighbors import NearestNeighbors as _cuNN  # type: ignore
        from cuml import PCA as _cuPCA  # type: ignore
        from cuml import UMAP as _cuUMAP  # type: ignore
        CUPY = cp
        CUML = {
            "KMeans": _cuKMeans,
            "NearestNeighbors": _cuNN,
            "PCA": _cuPCA,
            "UMAP": _cuUMAP,
        }
        CUML_AVAILABLE = True
        USE_CUML = True
        print("cuML/cuPy detected — GPU-accelerated reductions and clustering will be used where possible.")
    except Exception:
        CUML_AVAILABLE = False
        print("cuML/cuPy not available — falling back to CPU implementations.")






# Silhouette evaluations: raw embeddings (ground truth), PCA->50, UMAP->2D
raw_k, raw_score, _ = compute_best_silhouette(embeddings, k_min=2, k_max=10)
print(f"\nSilhouette (raw embeddings): best_k={raw_k}, score={raw_score:.4f}")

pca_points = None
umap_points = None
if USE_CUML:
    reduced = try_reduce_with_cuml(
        embeddings,
        n_components=3,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
    )
    if reduced is not None:
        pca_points, umap_points = reduced

if pca_points is None or umap_points is None:
    pca_points = reduce_pca(embeddings, n_components=3)
    umap_points = reduce_umap(
        embeddings,
        n_components=3,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
    )

# Additional pipeline: reduce embeddings to 50 dimensions with PCA, then UMAP -> 3D
# This can help UMAP focus on a compact subspace instead of the full embedding dims.
pca50_points = reduce_pca(embeddings, n_components=50)
umap_from_pca50 = reduce_umap(
    pca50_points,
    n_components=3,
    n_neighbors=UMAP_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
)

# Silhouette on PCA-50
p50_k, p50_score, _ = compute_best_silhouette(pca50_points, k_min=2, k_max=10)
print(f"Silhouette (PCA->50): best_k={p50_k}, score={p50_score:.4f}")

# UMAP 2D from raw embeddings for silhouette check
umap_2d_from_embeddings = reduce_umap(
    embeddings,
    n_components=2,
    n_neighbors=UMAP_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
)
u2_k, u2_score, _ = compute_best_silhouette(umap_2d_from_embeddings, k_min=2, k_max=10)
print(f"Silhouette (UMAP->2D from raw): best_k={u2_k}, score={u2_score:.4f}")

pca_fig = make_plot(pca_points, titles, "PCA (3D) - Title Embeddings")
umap_fig = make_plot(umap_points, titles, "UMAP (3D) - Title Embeddings")
umap50_fig = make_plot(umap_from_pca50, titles, "PCA-50 then UMAP (3D) - Title Embeddings")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pca_path = OUTPUT_DIR / "titles_pca_3d.html"
umap_path = OUTPUT_DIR / "titles_umap_3d.html"
pca50_umap_path = OUTPUT_DIR / "titles_pca50_umap_3d.html"
pca_fig.write_html(str(pca_path), include_plotlyjs="cdn")
umap_fig.write_html(str(umap_path), include_plotlyjs="cdn")
umap50_fig.write_html(str(pca50_umap_path), include_plotlyjs="cdn")
print(f"Loaded {len(titles)} titles from: {csv_path}")
print(f"Embedding device: {device}")
print(f"Saved PCA plot to: {pca_path}")
print(f"Saved UMAP plot to: {umap_path}")

# Intuitive representation outputs
print_repr_info('Raw embeddings (ground truth)', embeddings)
raw_k, raw_score, _ = compute_best_silhouette(embeddings, k_min=2, k_max=10)
print(f"Silhouette (raw embeddings): best_k={raw_k}, score={raw_score:.4f}")
run_clusterability_checks('Raw embeddings', embeddings)

print_repr_info('PCA (3D)', pca_points)
p_k, p_score, _ = compute_best_silhouette(pca_points, k_min=2, k_max=10)
print(f"Silhouette (PCA 3D): best_k={p_k}, score={p_score:.4f}")
run_clusterability_checks('PCA (3D)', pca_points)

print_repr_info('UMAP (3D)', umap_points)
u_k, u_score, _ = compute_best_silhouette(umap_points, k_min=2, k_max=10)
print(f"Silhouette (UMAP 3D): best_k={u_k}, score={u_score:.4f}")
run_clusterability_checks('UMAP (3D)', umap_points)

print_repr_info('PCA->50', pca50_points)
p50_k, p50_score, _ = compute_best_silhouette(pca50_points, k_min=2, k_max=10)
print(f"Silhouette (PCA->50): best_k={p50_k}, score={p50_score:.4f}")
run_clusterability_checks('PCA->50', pca50_points)

print_repr_info('PCA50->UMAP (3D)', umap_from_pca50)
u50_k, u50_score, _ = compute_best_silhouette(umap_from_pca50, k_min=2, k_max=10)
print(f"Silhouette (PCA50->UMAP 3D): best_k={u50_k}, score={u50_score:.4f}")
run_clusterability_checks('PCA50->UMAP (3D)', umap_from_pca50)

# UMAP 2D from raw embeddings for silhouette check (kept for compatibility)
print_repr_info('UMAP->2D (from raw)', umap_2d_from_embeddings)
u2_k, u2_score, _ = compute_best_silhouette(umap_2d_from_embeddings, k_min=2, k_max=10)
print(f"Silhouette (UMAP->2D from raw): best_k={u2_k}, score={u2_score:.4f}")
run_clusterability_checks('UMAP->2D (from raw)', umap_2d_from_embeddings)

if not NO_SHOW:
    pca_fig.show()
    umap_fig.show()
    umap50_fig.show()


