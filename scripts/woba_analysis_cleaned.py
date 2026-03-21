from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "plots" / "woba_analysis_cleaned"

# ── Control flags ───────────────────────────────────────────────────────────
YEARS = [2023,2024,2025]
# List of years to load and analyse.  Must be sequential and complete
# (i.e. no gaps).  Titles are derived from YEARS[0] and YEARS[-1].

ALL = False   # True  → all pitch types (4 rows × 5 cols)
             # False → top-2 rows only (10 highest-frequency pitch types)

FINE_GRID_SCALE = 4
# Controls the fine-grid bin size used as the background heatmap in the
# per-pitch-type four-panel plots.
#   1  → 1-inch bins  (default, finest resolution)
#   2  → 2-inch bins  (squares are 2× bigger, smoother, faster)
#   3  → 3-inch bins  (etc.)
# The coarse annotations are always drawn on top regardless of this value.

COARSE_DIVISIONS = 4
# Number of equal-width cells the strike zone is divided into along each axis
# for the coarse grid.  The grid extends beyond the zone to fill the view.
#   4  → 4×4 inside the zone (default, i.e. the classic 5-zone column layout)
#   5  → 5×5 inside the zone (finer coarse grid)
#   3  → 3×3 inside the zone (larger cells)
# NOTE: the visible x/y limits are always fixed at one 4-division cell width
# outside the strike zone (i.e. the same framing as the original default).
# Cells whose centres fall outside that window are rendered but not labelled.
# ────────────────────────────────────────────────────────────────────────────

# Keep strike-zone framing consistent with notebook work.
X_RANGE = (-1.5, 1.5)
Z_RANGE = (0.0, 5.0)
BIN_SIZE_FEET = 1.0 / 12.0  # 1-inch bins
WOBA_VMIN = 0.15
WOBA_VMAX = 0.45

THEORETICAL_STRIKE_ZONE = {
    "plate_x_min": -0.83,
    "plate_x_max": 0.83,
    "plate_z_min": 1.5,
    "plate_z_max": 3.5,
}

HIT_EVENTS = {"single", "double", "triple", "home_run"}
AB_EVENTS = {
    "single",
    "double",
    "triple",
    "home_run",
    "strikeout",
    "strikeout_double_play",
    "field_out",
    "force_out",
    "double_play",
    "grounded_into_double_play",
    "fielders_choice",
    "fielders_choice_out",
    "triple_play",
}
TOTAL_BASES = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4,
    "strikeout": 0,
    "strikeout_double_play": 0,
    "field_out": 0,
    "force_out": 0,
    "double_play": 0,
    "grounded_into_double_play": 0,
    "fielders_choice": 0,
    "fielders_choice_out": 0,
    "triple_play": 0,
}

PITCH_TYPE_NAMES = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "SL": "Slider",
    "CH": "Changeup",
    "FC": "Cutter",
    "ST": "Sweeper",
    "CU": "Curveball",
    "FS": "Splitter",
    "KC": "Knuckle Curve",
    "SV": "Slurve",
    "EP": "Eephus",
    "FA": "Fastball",
    "FO": "Forkball",
    "CS": "Slow Curve",
    "KN": "Knuckleball",
    "PO": "Pitch Out",
    "SC": "Screwball",
    "UN": "Unknown",
}


def build_edges(start: float, stop: float, step: float) -> np.ndarray:
    n_steps = int(round((stop - start) / step))
    return np.linspace(start, stop, n_steps + 1)


def build_divided_edges(
    zone_min: float,
    zone_max: float,
    divisions: int,
    axis_min: float,
    axis_max: float,
) -> np.ndarray:
    """Build edges where the strike zone is evenly divided, then extend outward."""
    width = zone_max - zone_min
    step = width / divisions
    base_edges = np.linspace(zone_min, zone_max, divisions + 1)

    edges = list(base_edges)
    while edges[0] - step > axis_min - 1e-9:
        edges.insert(0, edges[0] - step)
    while edges[-1] + step < axis_max + 1e-9:
        edges.append(edges[-1] + step)

    if edges[0] > axis_min:
        edges.insert(0, axis_min)
    if edges[-1] < axis_max:
        edges.append(axis_max)

    return np.array(edges, dtype=float)


def load_data() -> pd.DataFrame:
    cols = [
        "plate_x",
        "plate_z",
        "pitch_type",
        "events",
        "woba_value",
        "woba_denom",
    ]
    frames = []
    for year in YEARS:
        path = DATA_PATH / f"statcast_{year}.parquet"
        df = pd.read_parquet(path, columns=cols)
        df["year"] = year
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["plate_x", "plate_z"]).copy()
    return combined


def assign_bins(df: pd.DataFrame, x_edges: np.ndarray, z_edges: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["x_bin"] = pd.cut(out["plate_x"], bins=x_edges, labels=False, include_lowest=True)
    out["z_bin"] = pd.cut(out["plate_z"], bins=z_edges, labels=False, include_lowest=True)
    out = out.dropna(subset=["x_bin", "z_bin"]).copy()
    out["x_bin"] = out["x_bin"].astype(int)
    out["z_bin"] = out["z_bin"].astype(int)
    return out


def pivot_full(agg: pd.DataFrame, value_col: str, x_edges: np.ndarray, z_edges: np.ndarray) -> np.ndarray:
    n_x = len(x_edges) - 1
    n_z = len(z_edges) - 1
    matrix = np.full((n_z, n_x), np.nan, dtype=float)
    if agg.empty:
        return matrix
    matrix[agg["z_bin"].to_numpy(), agg["x_bin"].to_numpy()] = agg[value_col].to_numpy()
    return matrix


def compute_woba_matrix(df: pd.DataFrame, x_edges: np.ndarray, z_edges: np.ndarray) -> np.ndarray:
    data = df[df["woba_denom"].eq(1) & df["woba_value"].notna()].copy()
    data = assign_bins(data, x_edges, z_edges)

    if data.empty:
        return np.full((len(z_edges) - 1, len(x_edges) - 1), np.nan)

    agg = (
        data.groupby(["z_bin", "x_bin"], as_index=False)
        .agg(woba_sum=("woba_value", "sum"), denom=("woba_denom", "sum"))
    )
    agg["woba"] = agg["woba_sum"] / agg["denom"]
    return pivot_full(agg, "woba", x_edges, z_edges)


def compute_woba_count_matrix(df: pd.DataFrame, x_edges: np.ndarray, z_edges: np.ndarray) -> np.ndarray:
    """Return the number of PA (woba_denom==1) used to compute each wOBA cell."""
    data = df[df["woba_denom"].eq(1) & df["woba_value"].notna()].copy()
    data = assign_bins(data, x_edges, z_edges)
    if data.empty:
        return np.zeros((len(z_edges) - 1, len(x_edges) - 1), dtype=float)
    agg = data.groupby(["z_bin", "x_bin"], as_index=False).size().rename(columns={"size": "count"})
    counts = pivot_full(agg, "count", x_edges, z_edges)
    return np.nan_to_num(counts, nan=0.0)


def compute_ba_slg_matrices(
    df: pd.DataFrame, x_edges: np.ndarray, z_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = df[df["events"].isin(AB_EVENTS)].copy()
    data["is_hit"] = data["events"].isin(HIT_EVENTS).astype(int)
    data["total_bases"] = data["events"].map(TOTAL_BASES).fillna(0)
    data = assign_bins(data, x_edges, z_edges)

    if data.empty:
        shape = (len(z_edges) - 1, len(x_edges) - 1)
        return np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)

    agg = (
        data.groupby(["z_bin", "x_bin"], as_index=False)
        .agg(hits=("is_hit", "sum"), ab=("is_hit", "count"), tb=("total_bases", "sum"))
    )
    agg["ba"] = agg["hits"] / agg["ab"]
    agg["slg"] = agg["tb"] / agg["ab"]

    ba = pivot_full(agg, "ba", x_edges, z_edges)
    slg = pivot_full(agg, "slg", x_edges, z_edges)
    hits = pivot_full(agg, "hits", x_edges, z_edges)
    return ba, slg, hits


def compute_ab_count_matrix(df: pd.DataFrame, x_edges: np.ndarray, z_edges: np.ndarray) -> np.ndarray:
    data = df[df["events"].isin(AB_EVENTS)].copy()
    data = assign_bins(data, x_edges, z_edges)
    if data.empty:
        return np.zeros((len(z_edges) - 1, len(x_edges) - 1), dtype=float)

    agg = data.groupby(["z_bin", "x_bin"], as_index=False).size().rename(columns={"size": "count"})
    counts = pivot_full(agg, "count", x_edges, z_edges)
    return np.nan_to_num(counts, nan=0.0)


def compute_pitch_count_matrix(df: pd.DataFrame, x_edges: np.ndarray, z_edges: np.ndarray) -> np.ndarray:
    data = assign_bins(df, x_edges, z_edges)
    if data.empty:
        return np.zeros((len(z_edges) - 1, len(x_edges) - 1), dtype=float)

    agg = data.groupby(["z_bin", "x_bin"], as_index=False).size().rename(columns={"size": "count"})
    counts = pivot_full(agg, "count", x_edges, z_edges)
    return np.nan_to_num(counts, nan=0.0)


def _draw_zone(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.add_patch(
        Rectangle(
            (THEORETICAL_STRIKE_ZONE["plate_x_min"], THEORETICAL_STRIKE_ZONE["plate_z_min"]),
            THEORETICAL_STRIKE_ZONE["plate_x_max"] - THEORETICAL_STRIKE_ZONE["plate_x_min"],
            THEORETICAL_STRIKE_ZONE["plate_z_max"] - THEORETICAL_STRIKE_ZONE["plate_z_min"],
            linewidth=2.5,
            edgecolor="gray",
            facecolor="none",
            alpha=0.8,
        )
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')


def metric_limits(values: list[np.ndarray], lo_q: float = 0.05, hi_q: float = 0.95) -> tuple[float, float] | tuple[None, None]:
    flattened = [arr[np.isfinite(arr)] for arr in values]
    flattened = [arr for arr in flattened if arr.size > 0]
    if not flattened:
        return None, None
    combined = np.concatenate(flattened)
    return float(np.quantile(combined, lo_q)), float(np.quantile(combined, hi_q))


def _panel_count(all_flag: bool) -> int:
    """Return how many panels to plot based on ALL flag."""
    # ALL=True  → 4 rows × 5 cols = 20 panels
    # ALL=False → 2 rows × 5 cols = 10 panels
    return 20 if all_flag else 10


def plot_pitch_type_metric_grid_fine(
    metrics: dict,
    ordered_pitch_types: list[str],
    metric: str,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    output_path: Path,
    all_flag: bool = True,
) -> None:
    """Plot fine grid (1-inch) without annotations."""

    ncols = 5
    max_panels = _panel_count(all_flag)
    pitch_types_to_plot = ordered_pitch_types[:max_panels]
    n_panels = len(pitch_types_to_plot)
    nrows = int(np.ceil(n_panels / ncols))

    is_count = (metric == 'count')

    # ── Colour scale ──────────────────────────────────────────────────────
    if metric == 'woba':
        vmin = 0.0
        vmax = WOBA_VMAX
        vcenter = (WOBA_VMIN + WOBA_VMAX) / 2.0
        cmap = 'RdYlBu_r'
        cbar_label = 'wOBA'
        title_metric = 'wOBA'
        annotate_fmt = "{:.3f}"
    elif metric == 'ba':
        all_arrays = [metrics[pt]['ba'] for pt in pitch_types_to_plot if pt in metrics]
        _, vmax = metric_limits(all_arrays)
        all_finite = np.concatenate([a[np.isfinite(a)].ravel() for a in all_arrays if a.size > 0])
        vcenter = float(np.mean(all_finite)) if all_finite.size > 0 else (vmax or 0.3) / 2
        vmin = 0.0
        vmax = max(vmax or vcenter * 2, vcenter * 2)
        cmap = 'RdYlBu_r'
        cbar_label = 'Batting Average'
        title_metric = 'Batting Average'
        annotate_fmt = "{:.3f}"
    elif metric == 'slg':
        all_arrays = [metrics[pt]['slg'] for pt in pitch_types_to_plot if pt in metrics]
        _, vmax = metric_limits(all_arrays)
        all_finite = np.concatenate([a[np.isfinite(a)].ravel() for a in all_arrays if a.size > 0])
        vcenter = float(np.mean(all_finite)) if all_finite.size > 0 else (vmax or 0.4) / 2
        vmin = 0.0
        vmax = max(vmax or vcenter * 2, vcenter * 2)
        cmap = 'RdYlBu_r'
        cbar_label = 'Slugging Percentage'
        title_metric = 'Slugging'
        annotate_fmt = "{:.3f}"
    elif metric == 'count':
        vmin, vmax, vcenter = None, None, None
        cmap = 'YlOrRd'
        cbar_label = 'Number of Pitches'
        title_metric = 'Pitch Count'
        annotate_fmt = "{:,}"
    else:
        vmin, vmax, vcenter = None, None, None
        cmap = 'viridis'
        cbar_label = metric
        title_metric = metric
        annotate_fmt = "{:.3f}"

    # Extra bottom margin: count keeps per-panel cbars, others get shared cbar.
    # Increased from 0.09 → 0.14 so the n= / n_AB= labels clear the colorbar.
    bottom_margin = 0.02 if is_count else 0.14

    hspace = 0.35 if is_count else 0.175
    wspace = 0.10 if is_count else 0.05

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(25, 5 * nrows),
        gridspec_kw={'hspace': hspace, 'wspace': wspace},
    )
    axes = np.array(axes).flatten()

    for idx, ptype in enumerate(pitch_types_to_plot):
        if idx >= len(axes):
            break

        ax = axes[idx]

        if ptype not in metrics:
            ax.set_visible(False)
            continue

        matrix = metrics[ptype][metric]
        ab_matrix = metrics[ptype].get('ab_count')

        # N-count matrix used to annotate each cell for rate metrics
        if metric == 'woba':
            n_matrix = metrics[ptype].get('woba_count')
        elif metric in ('ba', 'slg'):
            n_matrix = ab_matrix
        else:
            n_matrix = None

        if is_count:
            panel_vals = matrix[matrix > 0]
            panel_vmin = 0
            panel_vmax = float(np.max(panel_vals)) if panel_vals.size > 0 else 1
            panel_norm = mcolors.Normalize(vmin=panel_vmin, vmax=panel_vmax)
        else:
            panel_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        im = ax.pcolormesh(
            x_edges, z_edges,
            np.ma.masked_invalid(matrix) if not is_count else matrix,
            cmap=cmap, shading='flat',
            norm=panel_norm,
        )

        # Individual colobar only for count
        if is_count:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        fine_xlim = (-1, 1)
        fine_zlim = (1.25, 3.75)
        _draw_zone(ax, xlim=fine_xlim, ylim=fine_zlim)

        # ── Annotations (only when bins are large enough to fit text) ──
        if FINE_GRID_SCALE >= 4:
            x_centers = x_edges[:-1] + np.diff(x_edges) / 2.0
            z_centers = z_edges[:-1] + np.diff(z_edges) / 2.0

            for zi, zc in enumerate(z_centers):
                for xi, xc in enumerate(x_centers):
                    # Skip any cell whose edges extend outside the view — text would bleed.
                    if not (x_edges[xi] >= fine_xlim[0] and x_edges[xi + 1] <= fine_xlim[1]
                            and z_edges[zi] >= fine_zlim[0] and z_edges[zi + 1] <= fine_zlim[1]):
                        continue
                    val = matrix[zi, xi]
                    if is_count:
                        if val == 0:
                            continue
                    else:
                        if not np.isfinite(val):
                            continue

                    if panel_norm is not None:
                        rgba = plt.colormaps[cmap](panel_norm(val))
                        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        text_color = "black" if lum > 0.65 else "white"
                    else:
                        text_color = "black"

                    if is_count:
                        count_val = int(val)
                        ab_val = int(ab_matrix[zi, xi]) if ab_matrix is not None and np.isfinite(ab_matrix[zi, xi]) else 0
                        ax.text(xc, zc + 0.06, f"{count_val:,}",
                                ha="center", va="center", fontsize=5.5,
                                color=text_color, fontweight='bold')
                        ax.text(xc, zc - 0.06, f"(AB: {ab_val:,})",
                                ha="center", va="center", fontsize=4.5,
                                color=text_color)
                    else:
                        n_val = int(n_matrix[zi, xi]) if (n_matrix is not None and np.isfinite(n_matrix[zi, xi])) else None
                        ax.text(xc, zc + 0.06, annotate_fmt.format(val),
                                ha="center", va="center", fontsize=6,
                                color=text_color, fontweight='bold')
                        if n_val is not None:
                            ax.text(xc, zc - 0.06, f"(N={n_val:,})",
                                    ha="center", va="center", fontsize=4.5,
                                    color=text_color)

        if ptype == 'all':
            panel_title = "All Pitches"
        else:
            panel_title = PITCH_TYPE_NAMES.get(ptype.upper(), ptype.upper())

        total = metrics[ptype]['total']
        total_ab = int(np.sum(metrics[ptype]['ab_count']))
        ax.set_title(
            f"{panel_title}\n(n={total:,}, n_AB={total_ab:,})",
            fontsize=11, fontweight='bold', pad=3,
        )

        # ── Larger tick labels ──────────────────────────────────────────
        ax.tick_params(axis='both', labelsize=10, length=4, width=1.2)

        if idx % ncols == 0:
            ax.set_ylabel("Plate Z (ft)", fontsize=10)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Plate X (ft)", fontsize=10)

    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    # ── Shared colorbar across the bottom (non-count only) ────────────
    if not is_count and vmin is not None:
        fig.subplots_adjust(top=0.96, bottom=bottom_margin, left=0.04, right=0.97)
        cbar_ax = fig.add_axes([0.10, 0.04, 0.80, 0.018])
        shared_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=shared_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(cbar_label, fontsize=11)
        cbar.ax.tick_params(labelsize=10)
    else:
        fig.subplots_adjust(top=0.96, bottom=bottom_margin, left=0.04, right=0.97)

    fine_bin_inches = int(round(BIN_SIZE_FEET * FINE_GRID_SCALE * 12))
    fig.suptitle(
        f"{title_metric} by Pitch Type and Location ({YEARS[0]}–{YEARS[-1]}) | "
        f"{fine_bin_inches}-inch grid, {n_panels} pitch types",
        fontsize=13, fontweight='bold', y=1.025, x=0.55
    )

    #plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path.name}")


def plot_pitch_type_metric_grid_coarse(
    metrics: dict,
    ordered_pitch_types: list[str],
    metric: str,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    output_path: Path,
    all_flag: bool = True,
) -> None:
    """Plot coarse grid (5×5) WITH annotations."""

    ncols = 5
    max_panels = _panel_count(all_flag)
    pitch_types_to_plot = ordered_pitch_types[:max_panels]
    n_panels = len(pitch_types_to_plot)
    nrows = int(np.ceil(n_panels / ncols))

    is_count = (metric == 'count')

    sz_x_min = THEORETICAL_STRIKE_ZONE["plate_x_min"]
    sz_x_max = THEORETICAL_STRIKE_ZONE["plate_x_max"]
    sz_z_min = THEORETICAL_STRIKE_ZONE["plate_z_min"]
    sz_z_max = THEORETICAL_STRIKE_ZONE["plate_z_max"]
    # xlim/zlim are fixed at one 4-division cell's margin regardless of COARSE_DIVISIONS.
    cell_width_x = (sz_x_max - sz_x_min) / 4.0
    cell_width_z = (sz_z_max - sz_z_min) / 4.0
    xlim = (sz_x_min - cell_width_x, sz_x_max + cell_width_x)
    zlim = (sz_z_min - cell_width_z, sz_z_max + cell_width_z)

    # ── Colour scale ──────────────────────────────────────────────────────
    if metric == 'woba':
        vmin = 0.0
        vmax = WOBA_VMAX
        vcenter = (WOBA_VMIN + WOBA_VMAX) / 2.0
        cmap = 'RdYlBu_r'
        cbar_label = 'wOBA'
        title_metric = 'wOBA'
        annotate_fmt = "{:.3f}"
    elif metric == 'ba':
        all_arrays = [metrics[pt]['ba'] for pt in pitch_types_to_plot if pt in metrics]
        _, vmax = metric_limits(all_arrays)
        all_finite = np.concatenate([a[np.isfinite(a)].ravel() for a in all_arrays if a.size > 0])
        vcenter = float(np.mean(all_finite)) if all_finite.size > 0 else (vmax or 0.3) / 2
        vmin = 0.0
        vmax = max(vmax or vcenter * 2, vcenter * 2)
        cmap = 'RdYlBu_r'
        cbar_label = 'Batting Average'
        title_metric = 'Batting Average'
        annotate_fmt = "{:.3f}"
    elif metric == 'slg':
        all_arrays = [metrics[pt]['slg'] for pt in pitch_types_to_plot if pt in metrics]
        _, vmax = metric_limits(all_arrays)
        all_finite = np.concatenate([a[np.isfinite(a)].ravel() for a in all_arrays if a.size > 0])
        vcenter = float(np.mean(all_finite)) if all_finite.size > 0 else (vmax or 0.4) / 2
        vmin = 0.0
        vmax = max(vmax or vcenter * 2, vcenter * 2)
        cmap = 'RdYlBu_r'
        cbar_label = 'Slugging Percentage'
        title_metric = 'Slugging'
        annotate_fmt = "{:.3f}"
    elif metric == 'count':
        vmin, vmax, vcenter = None, None, None
        cmap = 'YlOrRd'
        cbar_label = 'Number of Pitches'
        title_metric = 'Pitch Count'
        annotate_fmt = "{:,}"
    else:
        vmin, vmax, vcenter = None, None, None
        cmap = 'viridis'
        cbar_label = metric
        title_metric = metric
        annotate_fmt = "{:.3f}"

    # Extra bottom margin: count keeps per-panel cbars, others get shared cbar.
    # Increased from 0.09 → 0.14 so the n= / n_AB= labels clear the colorbar.
    bottom_margin = 0.02 if is_count else 0.14

    hspace = 0.35 if is_count else 0.175
    wspace = 0.10 if is_count else 0.05

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(25, 5 * nrows),
        gridspec_kw={'hspace': hspace, 'wspace': wspace},
    )
    axes = np.array(axes).flatten()

    for idx, ptype in enumerate(pitch_types_to_plot):
        if idx >= len(axes):
            break

        ax = axes[idx]

        if ptype not in metrics:
            ax.set_visible(False)
            continue

        matrix = metrics[ptype][metric]
        ab_matrix = metrics[ptype].get('ab_count')

        # N-count matrix used to annotate each cell for rate metrics
        if metric == 'woba':
            n_matrix = metrics[ptype].get('woba_count')
        elif metric in ('ba', 'slg'):
            n_matrix = ab_matrix
        else:
            n_matrix = None

        if is_count:
            panel_vals = matrix[matrix > 0]
            panel_vmin = 0
            panel_vmax = float(np.max(panel_vals)) if panel_vals.size > 0 else 1
            panel_norm = mcolors.Normalize(vmin=panel_vmin, vmax=panel_vmax)
        else:
            panel_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        im = ax.pcolormesh(
            x_edges, z_edges,
            np.ma.masked_invalid(matrix) if not is_count else matrix,
            cmap=cmap, shading='flat',
            norm=panel_norm,
        )

        if is_count:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        _draw_zone(ax, xlim=xlim, ylim=zlim)

        x_centers = x_edges[:-1] + np.diff(x_edges) / 2.0
        z_centers = z_edges[:-1] + np.diff(z_edges) / 2.0

        for zi, zc in enumerate(z_centers):
            for xi, xc in enumerate(x_centers):
                if not (xlim[0] <= xc <= xlim[1] and zlim[0] <= zc <= zlim[1]):
                    continue
                val = matrix[zi, xi]
                if is_count:
                    if val == 0:
                        continue
                else:
                    if not np.isfinite(val):
                        continue

                if panel_norm is not None:
                    rgba = plt.colormaps[cmap](panel_norm(val))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = "black" if lum > 0.65 else "white"
                else:
                    text_color = "black"

                if is_count:
                    count_val = int(val)
                    ab_val = int(ab_matrix[zi, xi]) if ab_matrix is not None and np.isfinite(ab_matrix[zi, xi]) else 0
                    ax.text(xc, zc + 0.06, f"{count_val:,}",
                            ha="center", va="center", fontsize=5.5,
                            color=text_color, fontweight='bold')
                    ax.text(xc, zc - 0.06, f"(AB: {ab_val:,})",
                            ha="center", va="center", fontsize=4.5,
                            color=text_color)
                else:
                    n_val = int(n_matrix[zi, xi]) if (n_matrix is not None and np.isfinite(n_matrix[zi, xi])) else None
                    ax.text(xc, zc + 0.06, annotate_fmt.format(val),
                            ha="center", va="center", fontsize=6,
                            color=text_color, fontweight='bold')
                    if n_val is not None:
                        ax.text(xc, zc - 0.06, f"(N={n_val:,})",
                                ha="center", va="center", fontsize=4.5,
                                color=text_color)

        if ptype == 'all':
            panel_title = "All Pitches"
        else:
            panel_title = PITCH_TYPE_NAMES.get(ptype.upper(), ptype.upper())

        total = metrics[ptype]['total']
        total_ab = int(np.sum(metrics[ptype]['ab_count']))
        ax.set_title(
            f"{panel_title}\n(n={total:,}, n_AB={total_ab:,})",
            fontsize=11, fontweight='bold', pad=3,
        )

        # ── Larger tick labels ──────────────────────────────────────────
        ax.tick_params(axis='both', labelsize=10, length=4, width=1.2)

        if idx % ncols == 0:
            ax.set_ylabel("Plate Z (ft)", fontsize=10)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Plate X (ft)", fontsize=10)

    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    # ── Shared colorbar across the bottom (non-count only) ────────────
    if not is_count and vmin is not None:
        fig.subplots_adjust(top=0.96, bottom=bottom_margin, left=0.04, right=0.97)
        cbar_ax = fig.add_axes([0.10, 0.04, 0.80, 0.018])
        shared_norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=shared_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(cbar_label, fontsize=11)
        cbar.ax.tick_params(labelsize=10)
    else:
        fig.subplots_adjust(top=0.96, bottom=bottom_margin, left=0.04, right=0.97)

    fig.suptitle(
        f"{title_metric} by Pitch Type and Location ({YEARS[0]}–{YEARS[-1]}) | "
        f"5×5 coarse grid, {n_panels} pitch types",
        fontsize=13, fontweight='bold', y=1.025, x=0.55
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path.name}")


def _annotate_coarse_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    n_matrix: np.ndarray | None,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    cmap: str,
    norm: mcolors.Normalize,
    is_count: bool,
    annotate_fmt: str,
    ab_matrix: np.ndarray | None = None,
) -> None:
    """Shared helper: draw cell-value (+ N= sub-label) annotations on a coarse panel.

    For count panels, an optional ab_matrix adds a second '(AB: XX)' line below
    the pitch count.  For rate panels, n_matrix adds an '(N=XX)' sub-label.
    """
    x_centers = x_edges[:-1] + np.diff(x_edges) / 2.0
    z_centers = z_edges[:-1] + np.diff(z_edges) / 2.0
    for zi, zc in enumerate(z_centers):
        for xi, xc in enumerate(x_centers):
            if not (xlim[0] <= xc <= xlim[1] and ylim[0] <= zc <= ylim[1]):
                continue
            val = matrix[zi, xi]
            if is_count:
                if val == 0:
                    continue
            else:
                if not np.isfinite(val):
                    continue

            rgba = plt.colormaps[cmap](norm(val))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "black" if lum > 0.65 else "white"

            if is_count:
                ab_val = (
                    int(ab_matrix[zi, xi])
                    if (ab_matrix is not None and np.isfinite(ab_matrix[zi, xi]))
                    else None
                )
                ax.text(xc, zc + 0.06, annotate_fmt.format(int(val)),
                        ha="center", va="center", fontsize=7,
                        color=text_color, fontweight='bold')
                if ab_val is not None:
                    ax.text(xc, zc - 0.07, f"(AB: {ab_val:,})",
                            ha="center", va="center", fontsize=5,
                            color=text_color)
            else:
                n_val = int(n_matrix[zi, xi]) if (n_matrix is not None and np.isfinite(n_matrix[zi, xi])) else None
                ax.text(xc, zc + 0.06, annotate_fmt.format(val),
                        ha="center", va="center", fontsize=7,
                        color=text_color, fontweight='bold')
                if n_val is not None:
                    ax.text(xc, zc - 0.07, f"(N={n_val:,})",
                            ha="center", va="center", fontsize=5,
                            color=text_color)


def plot_per_pitch_type(
    metrics: dict,
    ordered_pitch_types: list[str],
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    output_dir: Path,
    metrics_fine: dict | None = None,
    fine_x_edges: np.ndarray | None = None,
    fine_z_edges: np.ndarray | None = None,
    fine_suffix: str = "",
) -> None:
    """
    For each pitch type, produce a single 2×2 figure with four coarse-grid panels:
    top-left  → Pitch Count
    top-right → wOBA
    bottom-left  → Batting Average
    bottom-right → Slugging Percentage

    If metrics_fine / fine_x_edges / fine_z_edges are provided, the fine-grid
    heatmap is rendered first as an unlabelled background; the coarse-grid
    annotations are then drawn on top as usual.

    fine_suffix is appended to the output filename stem (e.g. "_fine_bg").

    Saved as  <output_dir>/per_pitch_type/<PTYPE>_four_panel{fine_suffix}.png
    """
    per_pt_dir = output_dir / "per_pitch_type"
    per_pt_dir.mkdir(parents=True, exist_ok=True)

    sz_x_min = THEORETICAL_STRIKE_ZONE["plate_x_min"]
    sz_x_max = THEORETICAL_STRIKE_ZONE["plate_x_max"]
    sz_z_min = THEORETICAL_STRIKE_ZONE["plate_z_min"]
    sz_z_max = THEORETICAL_STRIKE_ZONE["plate_z_max"]
    # xlim/zlim are fixed at one 4-division cell's margin regardless of COARSE_DIVISIONS.
    # Cells outside this window are rendered (no white gaps) but not annotated.
    cell_w_x = (sz_x_max - sz_x_min) / 4.0
    cell_w_z = (sz_z_max - sz_z_min) / 4.0
    xlim = (sz_x_min - cell_w_x, sz_x_max + cell_w_x)
    zlim = (sz_z_min - cell_w_z, sz_z_max + cell_w_z)

    panel_cfg = [
        # (key,    title,               cmap,        vmin,       vmax,       fmt,      is_count)
        ('count',  'Pitch Count',       'YlOrRd',    None,       None,       '{:,}',   True),
        ('woba',   'wOBA',              'RdBu_r',  WOBA_VMIN,  WOBA_VMAX,  '{:.3f}', False),
        ('ba',     'Batting Average',   'RdBu_r',  None,       None,       '{:.3f}', False),
        ('slg',    'Slugging %',        'RdBu_r',  None,       None,       '{:.3f}', False),
    ]

    # Pre-compute global vmin/vmax for ba and slg across all pitch types
    all_ba  = [metrics[pt]['ba']  for pt in ordered_pitch_types if pt in metrics]
    all_slg = [metrics[pt]['slg'] for pt in ordered_pitch_types if pt in metrics]
    ba_vmin,  ba_vmax  = metric_limits(all_ba)
    slg_vmin, slg_vmax = metric_limits(all_slg)

    # Patch in computed limits
    panel_cfg_resolved = []
    for key, title, cmap, vmin, vmax, fmt, is_count in panel_cfg:
        if key == 'ba':
            vmin, vmax = ba_vmin, ba_vmax
        elif key == 'slg':
            vmin, vmax = slg_vmin, slg_vmax
        panel_cfg_resolved.append((key, title, cmap, vmin, vmax, fmt, is_count))

    for ptype in ordered_pitch_types:
        if ptype not in metrics:
            continue

        m = metrics[ptype]
        ptype_label = "All_Pitches" if ptype == 'all' else ptype.upper()
        ptype_title = "All Pitches" if ptype == 'all' else PITCH_TYPE_NAMES.get(ptype.upper(), ptype.upper())

        fig, axes = plt.subplots(
            2, 2,
            figsize=(12, 11),
            gridspec_kw={'hspace': 0.25, 'wspace': 0.15},
        )
        axes_flat = axes.flatten()

        for ax, (key, title, cmap, vmin, vmax, fmt, is_count) in zip(axes_flat, panel_cfg_resolved):
            matrix = m[key]

            # Decide early whether we'll use the fine background, so the colour
            # scale can be derived from the correct matrix.
            use_fine = (
                metrics_fine is not None
                and fine_x_edges is not None
                and fine_z_edges is not None
                and ptype in metrics_fine
            )
            scale_matrix = metrics_fine[ptype][key] if use_fine else matrix
            scale_x_edges = fine_x_edges if use_fine else x_edges
            scale_z_edges = fine_z_edges if use_fine else z_edges

            if is_count:
                panel_vals = scale_matrix[scale_matrix > 0]
                pv_min = 0
                pv_max = float(np.max(panel_vals)) if panel_vals.size > 0 else 1
            else:
                x_centers = scale_x_edges[:-1] + np.diff(scale_x_edges) / 2.0
                z_centers = scale_z_edges[:-1] + np.diff(scale_z_edges) / 2.0
                visible_vals = np.array([
                    scale_matrix[zi, xi]
                    for zi, zc in enumerate(z_centers)
                    for xi, xc in enumerate(x_centers)
                    if xlim[0] <= xc <= xlim[1] and zlim[0] <= zc <= zlim[1]
                    and np.isfinite(scale_matrix[zi, xi])
                ])
                if visible_vals.size > 0:
                    if use_fine:
                        # Iterative 3-sigma clip: prevents sparse outlier cells
                        # from blowing out the colorbar range on the fine grid.
                        clipped = visible_vals.copy()
                        for _ in range(5):
                            mu = float(np.mean(clipped))
                            sig = float(np.std(clipped))
                            sig = sig if sig > 0 else 1e-6
                            clipped = clipped[np.abs(clipped - mu) <= 3 * sig]
                            if clipped.size == 0:
                                clipped = visible_vals.copy()
                                break
                        panel_mean = float(np.mean(clipped))
                        panel_std  = float(np.std(clipped))
                        panel_std  = panel_std if panel_std > 0 else 1e-6
                        pv_max = panel_mean + 3 * panel_std
                    else:
                        # Coarse grid: keep original mean ± 0.8*max_dev behaviour
                        panel_mean = float(np.mean(visible_vals))
                        max_dev = float(np.max(np.abs(visible_vals - panel_mean)))
                        max_dev = max_dev if max_dev > 0 else 1e-6
                        pv_max = panel_mean + 0.8 * max_dev
                    # Lower bound is always 0 (rate stats can't be negative).
                    # TwoSlopeNorm keeps white pinned at panel_mean regardless
                    # of how asymmetric the 0…mean…max range is.
                    pv_min = 0.0
                    vcenter = max(panel_mean, 1e-6)   # must be strictly > vmin
                    pv_max  = max(pv_max, vcenter + 1e-6)  # must be > vcenter
                else:
                    pv_min, pv_max = 0.0, (vmax if vmax is not None else 1.0)
                    vcenter = pv_max / 2.0

            if is_count:
                norm = mcolors.Normalize(vmin=pv_min, vmax=pv_max)
            else:
                norm = mcolors.TwoSlopeNorm(vmin=pv_min, vcenter=vcenter, vmax=pv_max)

            # ── Background: fine grid (unlabelled) if available; else coarse ──
            if use_fine:
                fine_matrix = metrics_fine[ptype][key]
                im = ax.pcolormesh(
                    fine_x_edges, fine_z_edges,
                    np.ma.masked_invalid(fine_matrix) if not is_count else fine_matrix,
                    cmap=cmap, shading='flat',
                    norm=norm,
                )
            else:
                im = ax.pcolormesh(
                    x_edges, z_edges,
                    np.ma.masked_invalid(matrix) if not is_count else matrix,
                    cmap=cmap, shading='flat',
                    norm=norm,
                )

            # N-count matrix for this panel
            if key == 'woba':
                n_mat = m.get('woba_count')
            elif key in ('ba', 'slg'):
                n_mat = m.get('ab_count')
            else:
                n_mat = None

            # AB matrix shown as a sub-label on the pitch-count panel
            ab_mat = m.get('ab_count') if key == 'count' else None

            if use_fine and FINE_GRID_SCALE >= 4:
                # Annotate the fine grid cells — identical logic to the
                # fine overview plots, with the full-cell-edges bounds check
                # so no text bleeds outside the panel.
                fine_matrix = metrics_fine[ptype][key]
                if key == 'woba':
                    fine_n_mat = metrics_fine[ptype].get('woba_count')
                elif key in ('ba', 'slg'):
                    fine_n_mat = metrics_fine[ptype].get('ab_count')
                else:
                    fine_n_mat = None
                fine_ab_mat = metrics_fine[ptype].get('ab_count') if key == 'count' else None

                fx_centers = fine_x_edges[:-1] + np.diff(fine_x_edges) / 2.0
                fz_centers = fine_z_edges[:-1] + np.diff(fine_z_edges) / 2.0

                for fzi, fzc in enumerate(fz_centers):
                    for fxi, fxc in enumerate(fx_centers):
                        # Only annotate cells whose edges are fully inside the view.
                        if not (fine_x_edges[fxi] >= xlim[0] and fine_x_edges[fxi + 1] <= xlim[1]
                                and fine_z_edges[fzi] >= zlim[0] and fine_z_edges[fzi + 1] <= zlim[1]):
                            continue
                        val = fine_matrix[fzi, fxi]
                        if is_count:
                            if val == 0:
                                continue
                        else:
                            if not np.isfinite(val):
                                continue

                        rgba = plt.colormaps[cmap](norm(val))
                        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        text_color = "black" if lum > 0.65 else "white"

                        if is_count:
                            ab_val = int(fine_ab_mat[fzi, fxi]) if fine_ab_mat is not None and np.isfinite(fine_ab_mat[fzi, fxi]) else 0
                            ax.text(fxc, fzc + 0.06, f"{int(val):,}",
                                    ha="center", va="center", fontsize=5.5,
                                    color=text_color, fontweight='bold')
                            ax.text(fxc, fzc - 0.06, f"(AB: {ab_val:,})",
                                    ha="center", va="center", fontsize=4.5,
                                    color=text_color)
                        else:
                            n_val = int(fine_n_mat[fzi, fxi]) if (fine_n_mat is not None and np.isfinite(fine_n_mat[fzi, fxi])) else None
                            ax.text(fxc, fzc + 0.06, fmt.format(val),
                                    ha="center", va="center", fontsize=6,
                                    color=text_color, fontweight='bold')
                            if n_val is not None:
                                ax.text(fxc, fzc - 0.06, f"(N={n_val:,})",
                                        ha="center", va="center", fontsize=4.5,
                                        color=text_color)
            elif not use_fine:
                _annotate_coarse_panel(
                    ax, matrix, n_mat,
                    x_edges, z_edges, xlim, zlim,
                    cmap, norm, is_count, fmt,
                    ab_matrix=ab_mat,
                )

            _draw_zone(ax, xlim=xlim, ylim=zlim)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            if not is_count:
                cbar.set_label(title, fontsize=9)

            ax.set_title(title, fontsize=12, fontweight='bold', pad=4)
            ax.tick_params(axis='both', labelsize=9)
            ax.set_ylabel("Plate Z (ft)", fontsize=9)
            ax.set_xlabel("Plate X (ft)", fontsize=9)

        total = m['total']
        total_ab = int(np.sum(m['ab_count']))
        fig.suptitle(
            f"{ptype_title}  |  {YEARS[0]}–{YEARS[-1]}\n"
            f"n={total:,} pitches   n_AB={total_ab:,}",
            fontsize=14, fontweight='bold', y=0.95, x=0.55,
        )

        out_path = per_pt_dir / f"{ptype_label}_four_panel{fine_suffix}.png"
        plt.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: per_pitch_type/{out_path.name}")


def subtract_visible_median(
    matrix: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    is_count: bool = False,
) -> tuple[np.ndarray, float]:
    """Return (matrix − median, median) where median is taken over visible, valid cells.

    'Visible' means the cell centre falls within xlim × ylim.
    'Valid' means finite for rate metrics, or > 0 for count metrics.
    Cells outside the visible window are left unchanged (they stay NaN / 0
    and won't be annotated anyway).
    """
    x_centers = x_edges[:-1] + np.diff(x_edges) / 2.0
    z_centers = z_edges[:-1] + np.diff(z_edges) / 2.0

    visible_vals = []
    for zi, zc in enumerate(z_centers):
        for xi, xc in enumerate(x_centers):
            if not (xlim[0] <= xc <= xlim[1] and ylim[0] <= zc <= ylim[1]):
                continue
            val = matrix[zi, xi]
            if is_count:
                if val > 0:
                    visible_vals.append(val)
            else:
                if np.isfinite(val):
                    visible_vals.append(val)

    if not visible_vals:
        return matrix.copy(), 0.0

    median_val = float(np.median(visible_vals))
    return matrix - median_val, median_val


def plot_per_pitch_type_deviation(
    metrics: dict,
    ordered_pitch_types: list[str],
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Identical layout to plot_per_pitch_type (2×2 coarse panels per pitch type),
    but each cell shows  value − median(visible cells)  instead of the raw value.
    A diverging colormap centred at 0 is used for all four panels.
    Saved as  <output_dir>/per_pitch_type/<PTYPE>_four_panel_deviation.png
    """
    per_pt_dir = output_dir / "per_pitch_type"
    per_pt_dir.mkdir(parents=True, exist_ok=True)

    sz_x_min = THEORETICAL_STRIKE_ZONE["plate_x_min"]
    sz_x_max = THEORETICAL_STRIKE_ZONE["plate_x_max"]
    sz_z_min = THEORETICAL_STRIKE_ZONE["plate_z_min"]
    sz_z_max = THEORETICAL_STRIKE_ZONE["plate_z_max"]
    # xlim/zlim are fixed at one 4-division cell's margin regardless of COARSE_DIVISIONS.
    cell_w_x = (sz_x_max - sz_x_min) / 4.0
    cell_w_z = (sz_z_max - sz_z_min) / 4.0
    xlim = (sz_x_min - cell_w_x, sz_x_max + cell_w_x)
    zlim = (sz_z_min - cell_w_z, sz_z_max + cell_w_z)

    # All deviation panels share the same diverging cmap; the scale is
    # symmetric around 0 (±max absolute deviation in the visible window).
    DEV_CMAP = 'RdBu_r'

    panel_cfg = [
        # (key,    title,                       fmt_fn,                              is_count)
        ('count',  'Pitch Count\n(− median)',   lambda v: f"{v:+,.0f}",              True),
        ('woba',   'wOBA\n(− median)',           lambda v: f"{v:+.3f}",               False),
        ('ba',     'Batting Average\n(− median)',lambda v: f"{v:+.3f}",               False),
        ('slg',    'Slugging %\n(− median)',     lambda v: f"{v:+.3f}",               False),
    ]

    for ptype in ordered_pitch_types:
        if ptype not in metrics:
            continue

        m = metrics[ptype]
        ptype_label = "All_Pitches" if ptype == 'all' else ptype.upper()
        ptype_title = "All Pitches" if ptype == 'all' else PITCH_TYPE_NAMES.get(ptype.upper(), ptype.upper())

        fig, axes = plt.subplots(
            2, 2,
            figsize=(12, 11),
            gridspec_kw={'hspace': 0.35, 'wspace': 0.25},
        )
        axes_flat = axes.flatten()

        for ax, (key, title, fmt_fn, is_count) in zip(axes_flat, panel_cfg):
            raw_matrix = m[key]

            dev_matrix, median_val = subtract_visible_median(
                raw_matrix, x_edges, z_edges, xlim, zlim, is_count=is_count,
            )

            # Symmetric colour scale centred at 0
            x_centers = x_edges[:-1] + np.diff(x_edges) / 2.0
            z_centers = z_edges[:-1] + np.diff(z_edges) / 2.0
            visible_devs = []
            for zi, zc in enumerate(z_centers):
                for xi, xc in enumerate(x_centers):
                    if not (xlim[0] <= xc <= xlim[1] and zlim[0] <= zc <= zlim[1]):
                        continue
                    v = dev_matrix[zi, xi]
                    if is_count:
                        if raw_matrix[zi, xi] > 0:
                            visible_devs.append(abs(v))
                    else:
                        if np.isfinite(v):
                            visible_devs.append(abs(v))

            max_abs = float(max(visible_devs)) if visible_devs else 1.0
            pv_min, pv_max = -max_abs, max_abs
            norm = mcolors.TwoSlopeNorm(vmin=pv_min, vcenter=0.0, vmax=pv_max)

            plot_matrix = np.ma.masked_invalid(dev_matrix) if not is_count else dev_matrix

            im = ax.pcolormesh(
                x_edges, z_edges, plot_matrix,
                cmap=DEV_CMAP, shading='flat',
                norm=norm,
            )

            # ── Cell annotations ──────────────────────────────────────────
            if key == 'woba':
                n_mat = m.get('woba_count')
            elif key in ('ba', 'slg'):
                n_mat = m.get('ab_count')
            else:
                n_mat = None
            ab_mat = m.get('ab_count') if key == 'count' else None

            for zi, zc in enumerate(z_centers):
                for xi, xc in enumerate(x_centers):
                    if not (xlim[0] <= xc <= xlim[1] and zlim[0] <= zc <= zlim[1]):
                        continue
                    dv = dev_matrix[zi, xi]
                    if is_count:
                        if raw_matrix[zi, xi] == 0:
                            continue
                    else:
                        if not np.isfinite(dv):
                            continue

                    rgba = plt.colormaps[DEV_CMAP](norm(dv))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = "black" if lum > 0.55 else "white"

                    ax.text(xc, zc + 0.06, fmt_fn(dv),
                            ha="center", va="center", fontsize=7,
                            color=text_color, fontweight='bold')

                    # Sub-label: AB count for pitch-count panel, N= for rate panels
                    if is_count and ab_mat is not None and np.isfinite(ab_mat[zi, xi]):
                        ax.text(xc, zc - 0.07, f"(AB: {int(ab_mat[zi, xi]):,})",
                                ha="center", va="center", fontsize=5, color=text_color)
                    elif not is_count and n_mat is not None and np.isfinite(n_mat[zi, xi]):
                        ax.text(xc, zc - 0.07, f"(N={int(n_mat[zi, xi]):,})",
                                ha="center", va="center", fontsize=5, color=text_color)

            _draw_zone(ax, xlim=xlim, ylim=zlim)

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

            median_str = f"{median_val:,.0f}" if is_count else f"{median_val:.3f}"
            ax.set_title(f"{title}\n(median = {median_str})", fontsize=11, fontweight='bold', pad=4)
            ax.tick_params(axis='both', labelsize=9)
            ax.set_ylabel("Plate Z (ft)", fontsize=9)
            ax.set_xlabel("Plate X (ft)", fontsize=9)

        total = m['total']
        total_ab = int(np.sum(m['ab_count']))
        fig.suptitle(
            f"{ptype_title}  |  {YEARS[0]}–{YEARS[-1]}  |  Deviation from median\n"
            f"n={total:,} pitches   n_AB={total_ab:,}",
            fontsize=14, fontweight='bold', y=0.95,x=0.55
        )

        out_path = per_pt_dir / f"{ptype_label}_four_panel_deviation.png"
        plt.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: per_pitch_type/{out_path.name}")


def _top_bottom_diff_section(
    metrics: dict,
    ordered_pitch_types: list[str],
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    grid_label: str,
    xlim: tuple[float, float],
    zlim: tuple[float, float],
    strict_edges: bool = False,
) -> list[str]:
    """
    Build the markdown lines for one grid's top-vs-bottom comparison
    (wOBA, BA, SLG).  Returns a list of markdown lines — the caller
    assembles and writes the file.

    strict_edges=False (coarse): a cell is included when its *center* falls
      within xlim/zlim — matching the coarse annotation loop.
    strict_edges=True (fine): a cell is included only when all four of its
      *edges* lie within xlim/zlim — matching the fine annotation loop which
      skips cells whose text would bleed outside the view.
    """
    x_centers = x_edges[:-1] + np.diff(x_edges) / 2.0
    z_centers = z_edges[:-1] + np.diff(z_edges) / 2.0

    # Indices within the visible window — edge-strict or center-based
    if strict_edges:
        x_idx = [
            xi for xi in range(len(x_centers))
            if x_edges[xi] >= xlim[0] and x_edges[xi + 1] <= xlim[1]
        ]
        z_idx = sorted([
            zi for zi in range(len(z_centers))
            if z_edges[zi] >= zlim[0] and z_edges[zi + 1] <= zlim[1]
        ])
    else:
        x_idx = [xi for xi, xc in enumerate(x_centers) if xlim[0] <= xc <= xlim[1]]
        z_idx = sorted([zi for zi, zc in enumerate(z_centers) if zlim[0] <= zc <= zlim[1]])

    if len(z_idx) < 2:
        return [f"\n> ⚠️ **{grid_label}**: fewer than 2 z-rows visible — skipped.\n"]

    top_zi = z_idx[-1]
    bot_zi = z_idx[0]
    top_z_range = f"{z_edges[top_zi]:.2f}–{z_edges[top_zi + 1]:.2f} ft"
    bot_z_range = f"{z_edges[bot_zi]:.2f}–{z_edges[bot_zi + 1]:.2f} ft"
    n_xcells = len(x_idx)

    metrics_cfg = [
        ('woba', 'wOBA', 'woba_count', '{:.3f}'),
        ('ba',   'BA',   'ab_count',   '{:.3f}'),
        ('slg',  'SLG',  'ab_count',   '{:.3f}'),
    ]

    lines: list[str] = [
        f"\n## {grid_label}\n",
        f"**Top row z-range:** {top_z_range}  ",
        f"**Bottom row z-range:** {bot_z_range}  ",
        f"**x-columns included:** {n_xcells} (centers within {xlim[0]:.2f} to {xlim[1]:.2f} ft)\n",
    ]

    top_cell_cols = " | ".join([f"Top C{i+1}" for i in range(n_xcells)])
    bot_cell_cols = " | ".join([f"Bot C{i+1}" for i in range(n_xcells)])
    sep_parts = ["---"] * (1 + n_xcells + 2 + n_xcells + 2 + 1)

    for metric_key, metric_label, count_key, fmt in metrics_cfg:
        lines.append(f"\n### {metric_label}\n")

        header = (
            f"| Pitch Type | {top_cell_cols} | Top Count | Top Avg "
            f"| {bot_cell_cols} | Bot Count | Bot Avg | Diff (Top−Bot) |"
        )
        sep = "| " + " | ".join(sep_parts) + " |"
        lines.append(header)
        lines.append(sep)

        for ptype in ordered_pitch_types:
            if ptype not in metrics:
                continue
            m = metrics[ptype]
            mat = m[metric_key]
            cnt = m[count_key]

            top_vals   = [mat[top_zi, xi] for xi in x_idx]
            top_counts = [cnt[top_zi, xi] for xi in x_idx]
            top_finite = [v for v in top_vals if np.isfinite(v)]
            top_avg    = float(np.mean(top_finite)) if top_finite else float('nan')
            top_total  = int(np.nansum([c for c in top_counts if np.isfinite(c)]))

            bot_vals   = [mat[bot_zi, xi] for xi in x_idx]
            bot_counts = [cnt[bot_zi, xi] for xi in x_idx]
            bot_finite = [v for v in bot_vals if np.isfinite(v)]
            bot_avg    = float(np.mean(bot_finite)) if bot_finite else float('nan')
            bot_total  = int(np.nansum([c for c in bot_counts if np.isfinite(c)]))

            diff = (top_avg - bot_avg) if (np.isfinite(top_avg) and np.isfinite(bot_avg)) else float('nan')

            ptype_label   = "All Pitches" if ptype == 'all' else PITCH_TYPE_NAMES.get(ptype.upper(), ptype.upper())
            top_cell_strs = " | ".join([fmt.format(v) if np.isfinite(v) else "—" for v in top_vals])
            bot_cell_strs = " | ".join([fmt.format(v) if np.isfinite(v) else "—" for v in bot_vals])
            top_avg_str   = fmt.format(top_avg) if np.isfinite(top_avg) else "—"
            bot_avg_str   = fmt.format(bot_avg) if np.isfinite(bot_avg) else "—"
            diff_str      = (("+" if diff > 0 else "") + fmt.format(diff)) if np.isfinite(diff) else "—"

            lines.append(
                f"| {ptype_label} | {top_cell_strs} | {top_total:,} | {top_avg_str} "
                f"| {bot_cell_strs} | {bot_total:,} | {bot_avg_str} | {diff_str} |"
            )

    return lines


def write_top_bottom_diff_table(
    metrics_coarse: dict,
    metrics_fine: dict,
    ordered_pitch_types: list[str],
    coarse_x: np.ndarray,
    coarse_z: np.ndarray,
    fine_x: np.ndarray,
    fine_z: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Write top_bottom_diff.md comparing the topmost vs bottommost visible row
    for both the coarse grid and the fine grid.

    Coarse visible window: same ±1-division margin used by the coarse plots.
    Fine visible window:   (-1, 1) × (1.25, 3.75) ft — matching fine overview plots.
    """
    sz_x_min = THEORETICAL_STRIKE_ZONE["plate_x_min"]
    sz_x_max = THEORETICAL_STRIKE_ZONE["plate_x_max"]
    sz_z_min = THEORETICAL_STRIKE_ZONE["plate_z_min"]
    sz_z_max = THEORETICAL_STRIKE_ZONE["plate_z_max"]
    cell_w_x = (sz_x_max - sz_x_min) / 4.0
    cell_w_z = (sz_z_max - sz_z_min) / 4.0
    coarse_xlim = (sz_x_min - cell_w_x, sz_x_max + cell_w_x)
    coarse_zlim = (sz_z_min - cell_w_z, sz_z_max + cell_w_z)

    # Fine visible limits match the fine overview plots
    fine_xlim = (-1.0, 1.0)
    fine_zlim = (1.25, 3.75)

    lines = [f"# Top vs Bottom Row — Zone Comparison ({YEARS[0]}–{YEARS[-1]})\n"]
    lines.append(
        "> Each section compares the **topmost** visible row of the grid against the "
        "**bottommost** visible row. Cell values are the raw metric per bin; "
        "Top/Bot Avg is the mean across all finite cells in that row; "
        "Diff = Top Avg − Bot Avg (positive means top-of-zone is higher).\n"
    )

    lines += _top_bottom_diff_section(
        metrics_coarse, ordered_pitch_types,
        coarse_x, coarse_z,
        f"Coarse Grid ({COARSE_DIVISIONS}×{COARSE_DIVISIONS} inside zone)",
        coarse_xlim, coarse_zlim,
        strict_edges=False,
    )
    if FINE_GRID_SCALE >= 4:
        lines += ["", "---"]
        lines += _top_bottom_diff_section(
            metrics_fine, ordered_pitch_types,
            fine_x, fine_z,
            f"Fine Grid ({int(round(BIN_SIZE_FEET * FINE_GRID_SCALE * 12))}-inch bins)",
            fine_xlim, fine_zlim,
            strict_edges=True,
        )

    out_path = output_dir / "top_bottom_diff.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: top_bottom_diff.md")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WOBA ANALYSIS - PITCH TYPE GRID PLOTS")
    print(f"ALL = {ALL}  →  plotting {'all pitch types (4 rows)' if ALL else 'top-10 pitch types (2 rows)'}")
    print("=" * 70)
    print(f"\nLoading {YEARS[0]}–{YEARS[-1]} Statcast data...")
    df = load_data()
    print(f"Loaded {len(df):,} pitches with location data")

    print("\nBuilding grid edges...")
    # fine_x/z: scaled by FINE_GRID_SCALE — used for both the overview
    # *_by_pitch_type_fine.png plots and the per-pitch-type fine-bg panels.
    fine_x = build_edges(X_RANGE[0], X_RANGE[1], BIN_SIZE_FEET * FINE_GRID_SCALE)
    fine_z = build_edges(Z_RANGE[0], Z_RANGE[1], BIN_SIZE_FEET * FINE_GRID_SCALE)
    per_pt_fine_x = fine_x
    per_pt_fine_z = fine_z
    coarse_x = build_divided_edges(
        THEORETICAL_STRIKE_ZONE["plate_x_min"],
        THEORETICAL_STRIKE_ZONE["plate_x_max"],
        divisions=COARSE_DIVISIONS,
        axis_min=X_RANGE[0],
        axis_max=X_RANGE[1],
    )
    coarse_z = build_divided_edges(
        THEORETICAL_STRIKE_ZONE["plate_z_min"],
        THEORETICAL_STRIKE_ZONE["plate_z_max"],
        divisions=COARSE_DIVISIONS,
        axis_min=Z_RANGE[0],
        axis_max=Z_RANGE[1],
    )
    fine_bin_inches = int(round(BIN_SIZE_FEET * FINE_GRID_SCALE * 12))
    print(f"Fine grid: {len(fine_x)-1} x-bins × {len(fine_z)-1} z-bins ({fine_bin_inches}-inch, FINE_GRID_SCALE={FINE_GRID_SCALE})")
    print(f"Coarse grid: {len(coarse_x)-1} x-bins × {len(coarse_z)-1} z-bins ({COARSE_DIVISIONS}×{COARSE_DIVISIONS} inside strike zone)")

    print("\nIdentifying pitch types by frequency...")
    pitch_type_counts = df['pitch_type'].value_counts()
    ordered_pitch_types = ['all'] + list(pitch_type_counts.index)
    max_panels = _panel_count(ALL)
    print(f"Found {len(ordered_pitch_types)} pitch types (including 'all'); plotting first {max_panels}")
    for i, ptype in enumerate(ordered_pitch_types[:10]):
        if ptype == 'all':
            print(f"  {i+1}. All Pitches: {len(df):,}")
        else:
            print(f"  {i+1}. {PITCH_TYPE_NAMES.get(ptype.upper(), ptype.upper())}: {pitch_type_counts[ptype]:,}")

    # Compute metrics for all pitch types (FINE — overview + per-pitch-type bg)
    print("\nComputing metrics for fine grid...")
    metrics_fine = {}
    for ptype in ordered_pitch_types[:max_panels]:
        subset = df if ptype == 'all' else df[df['pitch_type'] == ptype]
        ba, slg, _ = compute_ba_slg_matrices(subset, fine_x, fine_z)
        metrics_fine[ptype] = {
            'woba': compute_woba_matrix(subset, fine_x, fine_z),
            'woba_count': compute_woba_count_matrix(subset, fine_x, fine_z),
            'ba': ba,
            'slg': slg,
            'count': compute_pitch_count_matrix(subset, fine_x, fine_z),
            'ab_count': compute_ab_count_matrix(subset, fine_x, fine_z),
            'total': len(subset),
        }
    metrics_per_pt_fine = metrics_fine  # same grid, same data

    # Compute metrics for all pitch types (COARSE)
    print("Computing metrics for coarse grid...")
    metrics_coarse = {}
    for ptype in ordered_pitch_types[:max_panels]:
        subset = df if ptype == 'all' else df[df['pitch_type'] == ptype]
        ba, slg, _ = compute_ba_slg_matrices(subset, coarse_x, coarse_z)
        metrics_coarse[ptype] = {
            'woba': compute_woba_matrix(subset, coarse_x, coarse_z),
            'woba_count': compute_woba_count_matrix(subset, coarse_x, coarse_z),
            'ba': ba,
            'slg': slg,
            'count': compute_pitch_count_matrix(subset, coarse_x, coarse_z),
            'ab_count': compute_ab_count_matrix(subset, coarse_x, coarse_z),
            'total': len(subset),
        }

    # Generate fine plots
    print("\n" + "=" * 70)
    print("GENERATING FINE GRID PLOTS (no annotations)")
    print("=" * 70)
    for metric in ['woba', 'ba', 'slg', 'count']:
        print(f"\nGenerating {metric} fine grid...")
        plot_pitch_type_metric_grid_fine(
            metrics_fine, ordered_pitch_types[:max_panels], metric,
            fine_x, fine_z,
            OUTPUT_DIR / f"{metric}_by_pitch_type_fine.png",
            all_flag=ALL,
        )

    # Generate coarse plots
    print("\n" + "=" * 70)
    print("GENERATING COARSE GRID PLOTS (with annotations)")
    print("=" * 70)
    for metric in ['woba', 'ba', 'slg', 'count']:
        print(f"\nGenerating {metric} coarse grid...")
        plot_pitch_type_metric_grid_coarse(
            metrics_coarse, ordered_pitch_types[:max_panels], metric,
            coarse_x, coarse_z,
            OUTPUT_DIR / f"{metric}_by_pitch_type_coarse.png",
            all_flag=ALL,
        )

    # Generate per-pitch-type four-panel plots (coarse only)
    print("\n" + "=" * 70)
    print("GENERATING PER-PITCH-TYPE FOUR-PANEL PLOTS (coarse only)")
    print("=" * 70)
    plot_per_pitch_type(
        metrics_coarse,
        ordered_pitch_types[:max_panels],
        coarse_x, coarse_z,
        OUTPUT_DIR,
    )

    # Generate per-pitch-type four-panel plots (fine background + coarse annotations)
    print("\n" + "=" * 70)
    print("GENERATING PER-PITCH-TYPE FOUR-PANEL PLOTS (fine background)")
    print("=" * 70)
    plot_per_pitch_type(
        metrics_coarse,
        ordered_pitch_types[:max_panels],
        coarse_x, coarse_z,
        OUTPUT_DIR,
        metrics_fine=metrics_per_pt_fine,
        fine_x_edges=per_pt_fine_x,
        fine_z_edges=per_pt_fine_z,
        fine_suffix="_fine_bg",
    )

    # # Generate per-pitch-type deviation plots
    # print("\n" + "=" * 70)
    # print("GENERATING PER-PITCH-TYPE DEVIATION PLOTS")
    # print("=" * 70)
    # plot_per_pitch_type_deviation(
    #     metrics_coarse,
    #     ordered_pitch_types[:max_panels],
    #     coarse_x, coarse_z,
    #     OUTPUT_DIR,
    # )

    # Generate top/bottom row comparison table
    print("\n" + "=" * 70)
    print("GENERATING TOP/BOTTOM ZONE DIFF TABLE")
    print("=" * 70)
    write_top_bottom_diff_table(
        metrics_coarse,
        metrics_fine,
        ordered_pitch_types[:max_panels],
        coarse_x, coarse_z,
        fine_x, fine_z,
        OUTPUT_DIR,
    )

    print("\n" + "=" * 70)
    print(f"ALL PLOTS SAVED TO: {OUTPUT_DIR}")
    print("=" * 70)
    print("\nGenerated files:")
    for metric in ['woba', 'ba', 'slg', 'count']:
        print(f"  - {metric}_by_pitch_type_fine.png")
        print(f"  - {metric}_by_pitch_type_coarse.png")
    print(f"  - per_pitch_type/<PTYPE>_four_panel.png          ({max_panels} files)")
    print(f"  - per_pitch_type/<PTYPE>_four_panel_deviation.png ({max_panels} files)")
    print(f"  - top_bottom_diff.md")
    print("\nDone!")


if __name__ == "__main__":
    main()