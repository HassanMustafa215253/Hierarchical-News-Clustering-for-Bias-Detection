import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed CSV titles and visualize in 3D with PCA and UMAP."
    )
    parser.add_argument("--csv", type=Path, default=None, help="Path to train.csv")
    parser.add_argument(
        "--model",
        default="BAAI/bge-large-en-v1.5",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Embedding device selection",
    )
    parser.add_argument(
        "--use-cuml",
        action="store_true",
        help="Use RAPIDS cuML for GPU PCA/UMAP if available",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive windows",
    )
    return parser.parse_args()



args = parse_args()
csv_path = resolve_csv_path(args.csv)
device = select_device(args.device)

titles = load_titles_from_csv(csv_path)
if not titles:
    raise ValueError("No valid titles found in the CSV file.")

embeddings = embed_titles(titles, args.model, args.batch_size, device)

pca_points = None
umap_points = None
if args.use_cuml:
    reduced = try_reduce_with_cuml(
        embeddings,
        n_components=3,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    if reduced is not None:
        pca_points, umap_points = reduced

if pca_points is None or umap_points is None:
    pca_points = reduce_pca(embeddings, n_components=3)
    umap_points = reduce_umap(
        embeddings,
        n_components=3,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )

pca_fig = make_plot(pca_points, titles, "PCA (3D) - Title Embeddings")
umap_fig = make_plot(umap_points, titles, "UMAP (3D) - Title Embeddings")

args.output_dir.mkdir(parents=True, exist_ok=True)
pca_path = args.output_dir / "titles_pca_3d.html"
umap_path = args.output_dir / "titles_umap_3d.html"
pca_fig.write_html(str(pca_path), include_plotlyjs="cdn")
umap_fig.write_html(str(umap_path), include_plotlyjs="cdn")

print(f"Loaded {len(titles)} titles from: {csv_path}")
print(f"Embedding device: {device}")
print(f"Saved PCA plot to: {pca_path}")
print(f"Saved UMAP plot to: {umap_path}")

if not args.no_show:
    pca_fig.show()
    umap_fig.show()


