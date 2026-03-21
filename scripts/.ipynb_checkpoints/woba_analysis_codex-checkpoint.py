from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "plots" / "woba_analysis_codex"

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
}


def build_edges(start: float, stop: float, step: float) -> np.ndarray:
    # Include the right endpoint exactly for pd.cut bins.
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

    # Ensure the extremes cover the range boundaries.
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
    for year in [2023, 2024, 2025]:
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


def compute_ba_slg_matrices(
    df: pd.DataFrame, x_edges: np.ndarray, z_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    data = df[df["events"].isin(AB_EVENTS)].copy()
    data["is_hit"] = data["events"].isin(HIT_EVENTS).astype(int)
    data["total_bases"] = data["events"].map(TOTAL_BASES).fillna(0)
    data = assign_bins(data, x_edges, z_edges)

    if data.empty:
        shape = (len(z_edges) - 1, len(x_edges) - 1)
        return np.full(shape, np.nan), np.full(shape, np.nan)

    agg = (
        data.groupby(["z_bin", "x_bin"], as_index=False)
        .agg(hits=("is_hit", "sum"), ab=("is_hit", "count"), tb=("total_bases", "sum"))
    )
    agg["ba"] = agg["hits"] / agg["ab"]
    agg["slg"] = agg["tb"] / agg["ab"]

    ba = pivot_full(agg, "ba", x_edges, z_edges)
    slg = pivot_full(agg, "slg", x_edges, z_edges)
    return ba, slg


def compute_pitch_count_matrix(df: pd.DataFrame, x_edges: np.ndarray, z_edges: np.ndarray) -> np.ndarray:
    data = assign_bins(df, x_edges, z_edges)
    if data.empty:
        return np.zeros((len(z_edges) - 1, len(x_edges) - 1), dtype=float)

    agg = data.groupby(["z_bin", "x_bin"], as_index=False).size().rename(columns={"size": "count"})
    counts = pivot_full(agg, "count", x_edges, z_edges)
    return np.nan_to_num(counts, nan=0.0)


def compute_metric_set(
    subset: pd.DataFrame,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    ba, slg = compute_ba_slg_matrices(subset, x_edges, z_edges)
    return {
        "woba": compute_woba_matrix(subset, x_edges, z_edges),
        "ba": ba,
        "slg": slg,
        "count": compute_pitch_count_matrix(subset, x_edges, z_edges),
    }


def _draw_zone(ax: plt.Axes) -> None:
    ax.add_patch(
        Rectangle(
            (THEORETICAL_STRIKE_ZONE["plate_x_min"], THEORETICAL_STRIKE_ZONE["plate_z_min"]),
            THEORETICAL_STRIKE_ZONE["plate_x_max"] - THEORETICAL_STRIKE_ZONE["plate_x_min"],
            THEORETICAL_STRIKE_ZONE["plate_z_max"] - THEORETICAL_STRIKE_ZONE["plate_z_min"],
            linewidth=1.6,
            edgecolor="black",
            facecolor="none",
        )
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.axhline(2.5, color="black", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_xlim(X_RANGE)
    ax.set_ylim(Z_RANGE)


def _plot_grid(
    matrix: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = False,
    annotate_fmt: str = "{:.3f}",
    figsize: tuple[float, float] = (16, 12),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    mesh = ax.pcolormesh(
        x_edges,
        z_edges,
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
    )

    _draw_zone(ax)
    ax.set_xlabel("Plate X (ft)")
    ax.set_ylabel("Plate Z (ft)")
    ax.set_title(title)

    if annotate:
        if vmin is not None and vmax is not None and vmin < vmax:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            finite_vals = matrix[np.isfinite(matrix)]
            mn = float(np.nanmin(finite_vals)) if finite_vals.size else 0.0
            mx = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
            norm = mcolors.Normalize(vmin=mn, vmax=mx if mx > mn else mn + 1e-6)

        x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2.0
        z_centers = z_edges[:-1] + (z_edges[1] - z_edges[0]) / 2.0
        for zi, zc in enumerate(z_centers):
            for xi, xc in enumerate(x_centers):
                val = matrix[zi, xi]
                if not np.isfinite(val):
                    continue
                rgba = mesh.cmap(norm(val))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "black" if lum > 0.65 else "white"
                ax.text(xc, zc, annotate_fmt.format(val),
                        ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_count_grid(
    counts: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    out_path: Path,
    *,
    title: str,
    figsize: tuple[float, float] = (16, 12),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    mesh = ax.pcolormesh(
        x_edges,
        z_edges,
        counts,
        cmap="YlOrRd",
        shading="flat",
    )

    _draw_zone(ax)
    ax.set_xlabel("Plate X (ft)")
    ax.set_ylabel("Plate Z (ft)")
    ax.set_title(title)

    x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2.0
    z_centers = z_edges[:-1] + (z_edges[1] - z_edges[0]) / 2.0
    vmax = float(np.nanmax(counts)) if counts.size else 0.0

    for zi, zc in enumerate(z_centers):
        for xi, xc in enumerate(x_centers):
            val = int(counts[zi, xi])
            if val <= 0:
                continue
            color = "white" if (vmax > 0 and val > vmax * 0.45) else "black"
            ax.text(xc, zc, str(val), ha="center", va="center", fontsize=4, color=color)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Number of pitches")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def metric_limits(values: list[np.ndarray], lo_q: float = 0.05, hi_q: float = 0.95) -> tuple[float, float] | tuple[None, None]:
    flattened = [arr[np.isfinite(arr)] for arr in values]
    flattened = [arr for arr in flattened if arr.size > 0]
    if not flattened:
        return None, None
    combined = np.concatenate(flattened)
    return float(np.quantile(combined, lo_q)), float(np.quantile(combined, hi_q))


def collect_metric_arrays(metrics: dict[str, dict[str, np.ndarray]], metric_key: str) -> list[np.ndarray]:
    arrs = []
    for entry in metrics.values():
        arr = entry[metric_key]
        if np.isfinite(arr).any():
            arrs.append(arr)
    return arrs


def plot_metric_grid_figure(
    metric: str,
    matrices: dict[str, np.ndarray],
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    title: str,
    out_path: Path,
    vmin: float | None,
    vmax: float | None,
    *,
    annotate: bool = False,
    annotate_fmt: str = "{:.3f}",
    cmap: str = "RdYlBu_r",
    figsize_x: float = 18.0,
    figsize_y: float = 16.0,
    hspace: float = 0.35,
    wspace: float = 0.25,
    dpi: int = 160,
) -> None:
    labels = ["all"] + sorted(k for k in matrices.keys() if k != "all")
    n = len(labels)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_x, figsize_y),
                             gridspec_kw={"hspace": hspace, "wspace": wspace})
    axes = axes.flatten()

    if vmin is not None and vmax is not None and vmin < vmax:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    im = None
    for idx, label in enumerate(labels):
        ax = axes[idx]
        matrix = matrices.get(label)
        if matrix is not None and np.isfinite(matrix).any():
            im = ax.pcolormesh(
                x_edges,
                z_edges,
                np.ma.masked_invalid(matrix),
                cmap=cmap,
                shading="flat",
                vmin=vmin,
                vmax=vmax,
            )
            if annotate:
                x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2.0
                z_centers = z_edges[:-1] + (z_edges[1] - z_edges[0]) / 2.0
                for zi, zc in enumerate(z_centers):
                    for xi, xc in enumerate(x_centers):
                        val = matrix[zi, xi]
                        if not np.isfinite(val):
                            continue
                        use_norm = norm if norm is not None else mcolors.Normalize(
                            vmin=np.nanmin(matrix[np.isfinite(matrix)]) if np.isfinite(matrix).any() else 0,
                            vmax=np.nanmax(matrix[np.isfinite(matrix)]) if np.isfinite(matrix).any() else 1,
                        )
                        rgba = im.cmap(use_norm(val))
                        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        text_color = "black" if lum > 0.65 else "white"
                        ax.text(xc, zc, annotate_fmt.format(val),
                                ha="center", va="center", fontsize=2.5, color=text_color)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")

        _draw_zone(ax)

        title_label = "All pitches" if label == "all" else PITCH_TYPE_NAMES.get(label.upper(), label.upper())
        ax.set_title(title_label, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=7)

        if idx % ncols == 0:
            ax.set_ylabel("Plate Z (ft)")
        if idx >= n - ncols:
            ax.set_xlabel("Plate X (ft)")

    for idx in range(n, nrows * ncols):
        axes[idx].set_visible(False)

    if im is not None:
        cbar_ax = fig.add_axes([0.2, 0.01, 0.6, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(title)

    fig.suptitle(f"{title} (2023-2025)", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 2023-2025 Statcast data...")
    df = load_data()
    print(f"Loaded {len(df):,} pitches with location data")

    fine_x = build_edges(X_RANGE[0], X_RANGE[1], BIN_SIZE_FEET)
    fine_z = build_edges(Z_RANGE[0], Z_RANGE[1], BIN_SIZE_FEET)
    coarse_x = build_divided_edges(
        THEORETICAL_STRIKE_ZONE["plate_x_min"],
        THEORETICAL_STRIKE_ZONE["plate_x_max"],
        divisions=4,
        axis_min=X_RANGE[0],
        axis_max=X_RANGE[1],
    )
    coarse_z = build_divided_edges(
        THEORETICAL_STRIKE_ZONE["plate_z_min"],
        THEORETICAL_STRIKE_ZONE["plate_z_max"],
        divisions=4,
        axis_min=Z_RANGE[0],
        axis_max=Z_RANGE[1],
    )
    print(
        "Using 1-inch bins:",
        f"{len(fine_x)-1} x-bins and {len(fine_z)-1} z-bins",
    )
    print(
        "4×4 bins over strike zone:",
        f"{len(coarse_x)-1} x-bins and {len(coarse_z)-1} z-bins after extension",
    )

    keys = ["all"] + sorted(df["pitch_type"].dropna().unique())

    configs = [
        {
            "name": "fine",
            "label": "1-inch bins",
            "x_edges": fine_x,
            "z_edges": fine_z,
            "root": OUTPUT_DIR,
            "grid_root": OUTPUT_DIR / "pitch_types_fine",
            "grid_suffix": "",
            "pitch_types_dir": OUTPUT_DIR / "pitch_types_fine",
            "annotate": True,
            "figsize_x": 18.0,
            "figsize_y": 16.0,
            "hspace": 0.35,
            "wspace": 0.25,
            "dpi": 160,
            "single_figsize": (16, 12),
        },
        {
            "name": "coarse4x4",
            "label": "4×4 strike-zone bins",
            "x_edges": coarse_x,
            "z_edges": coarse_z,
            "root": OUTPUT_DIR / "coarse4x4",
            "grid_root": OUTPUT_DIR / "coarse4x4",
            "grid_suffix": "_coarse4x4",
            "pitch_types_dir": OUTPUT_DIR / "coarse4x4" / "pitch_types_coarse",
            "annotate": True,
            "figsize_x": 18.0,
            "figsize_y": 16.0,
            "hspace": 0.35,
            "wspace": 0.25,
            "dpi": 160,
            "single_figsize": (16, 12),
        },
    ]

    metrics_by_config: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for config in configs:
        config_root = config["root"]
        config_root.mkdir(parents=True, exist_ok=True)
        config["pitch_types_dir"].mkdir(parents=True, exist_ok=True)
        config_metrics: dict[str, dict[str, np.ndarray]] = {}
        for key in keys:
            subset = df if key == "all" else df[df["pitch_type"] == key]
            if subset.empty:
                continue
            config_metrics[key] = compute_metric_set(subset, config["x_edges"], config["z_edges"])
        metrics_by_config[config["name"]] = config_metrics

    print("Saving combined metric grids...")
    for config in configs:
        matrices = metrics_by_config[config["name"]]
        woba_min, woba_max = WOBA_VMIN, WOBA_VMAX
        ba_min, ba_max = metric_limits(collect_metric_arrays(matrices, "ba"))
        slg_min, slg_max = metric_limits(collect_metric_arrays(matrices, "slg"))

        root = config["root"]
        grid_root = config["grid_root"]
        grid_suffix = config["grid_suffix"]
        annotate = config["annotate"]

        plot_metric_grid_figure(
            "wOBA",
            {k: v["woba"] for k, v in matrices.items()},
            config["x_edges"],
            config["z_edges"],
            title="wOBA",
            out_path=grid_root / f"woba_grid_all_types{grid_suffix}.png",
            vmin=woba_min,
            vmax=woba_max,
            annotate=annotate,
            figsize_x=config["figsize_x"],
            figsize_y=config["figsize_y"],
            hspace=config["hspace"],
            wspace=config["wspace"],
            dpi=config["dpi"],
        )

        if ba_min is not None and ba_max is not None:
            plot_metric_grid_figure(
                "BA",
                {k: v["ba"] for k, v in matrices.items()},
                config["x_edges"],
                config["z_edges"],
                title="Batting Average",
                out_path=grid_root / f"ba_grid_all_types{grid_suffix}.png",
                vmin=ba_min,
                vmax=ba_max,
                annotate=annotate,
                figsize_x=config["figsize_x"],
                figsize_y=config["figsize_y"],
                hspace=config["hspace"],
                wspace=config["wspace"],
                dpi=config["dpi"],
            )

        if slg_min is not None and slg_max is not None:
            plot_metric_grid_figure(
                "SLG",
                {k: v["slg"] for k, v in matrices.items()},
                config["x_edges"],
                config["z_edges"],
                title="Slugging Percentage",
                out_path=grid_root / f"slg_grid_all_types{grid_suffix}.png",
                vmin=slg_min,
                vmax=slg_max,
                annotate=annotate,
                figsize_x=config["figsize_x"],
                figsize_y=config["figsize_y"],
                hspace=config["hspace"],
                wspace=config["wspace"],
                dpi=config["dpi"],
            )

    print("Saving per-type outputs...")
    for config in configs:
        matrices = metrics_by_config[config["name"]]
        ba_min, ba_max = metric_limits(collect_metric_arrays(matrices, "ba"))
        slg_min, slg_max = metric_limits(collect_metric_arrays(matrices, "slg"))
        config_root = config["root"]
        pitch_dir = config["pitch_types_dir"]

        for key, metrics in matrices.items():
            target_dir = pitch_dir if key == "all" else pitch_dir / key.lower()
            target_dir.mkdir(parents=True, exist_ok=True)

            title_prefix = "All pitches" if key == "all" else PITCH_TYPE_NAMES.get(key, key)
            if np.isfinite(metrics["woba"]).any():
                _plot_grid(
                    metrics["woba"],
                    config["x_edges"],
                    config["z_edges"],
                    title=f"{title_prefix} wOBA ({config['label']})",
                    cbar_label="wOBA",
                    out_path=target_dir / f"woba_heatmap_2023-25_{config['name']}_{key.lower()}.png",
                    cmap="RdYlBu_r",
                    vmin=WOBA_VMIN,
                    vmax=WOBA_VMAX,
                    annotate=config["annotate"],
                    figsize=config["single_figsize"],
                )
            if ba_min is not None and ba_max is not None and np.isfinite(metrics["ba"]).any():
                _plot_grid(
                    metrics["ba"],
                    config["x_edges"],
                    config["z_edges"],
                    title=f"{title_prefix} BA ({config['label']})",
                    cbar_label="Batting Average",
                    out_path=target_dir / f"ba_heatmap_2023-25_{config['name']}_{key.lower()}.png",
                    cmap="RdYlBu_r",
                    vmin=ba_min,
                    vmax=ba_max,
                    annotate=config["annotate"],
                    figsize=config["single_figsize"],
                )
            if slg_min is not None and slg_max is not None and np.isfinite(metrics["slg"]).any():
                _plot_grid(
                    metrics["slg"],
                    config["x_edges"],
                    config["z_edges"],
                    title=f"{title_prefix} SLG ({config['label']})",
                    cbar_label="Slugging Percentage",
                    out_path=target_dir / f"slg_heatmap_2023-25_{config['name']}_{key.lower()}.png",
                    cmap="RdYlBu_r",
                    vmin=slg_min,
                    vmax=slg_max,
                    annotate=config["annotate"],
                    figsize=config["single_figsize"],
                )

            if metrics["count"].sum() > 0:
                _plot_count_grid(
                    metrics["count"],
                    config["x_edges"],
                    config["z_edges"],
                    out_path=target_dir / f"pitch_count_heatmap_2023-25_{config['name']}_{key.lower()}.png",
                    title=f"{title_prefix} pitch count ({config['label']})",
                    figsize=config["single_figsize"],
                )

    print(f"Done. Plots saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
