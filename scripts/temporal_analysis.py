import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import os

DATA_DIR = "../data"
PLOTS_DIR = "../plots/temporal_analysis"

os.makedirs(PLOTS_DIR, exist_ok=True)

THEORETICAL_STRIKE_ZONE = {
    'plate_x_min': -0.83,
    'plate_x_max': 0.83,
    'plate_z_min': 1.5,
    'plate_z_max': 3.5
}

counts = [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2),
    (3, 0), (3, 1), (3, 2)
]

hitter_to_pitcher = {
    (0, 0): 0.5, (0, 1): 0.25, (0, 2): 0.0,
    (1, 0): 0.58, (1, 1): 0.33, (1, 2): 0.08,
    (2, 0): 0.67, (2, 1): 0.42, (2, 2): 0.17,
    (3, 0): 0.83, (3, 1): 0.75, (3, 2): 0.42
}


def load_data_for_year(year):
    path = os.path.join(DATA_DIR, f"statcast_{year}.parquet")
    df = pd.read_parquet(path)
    df['year'] = year
    df = df.dropna(subset=['plate_x', 'plate_z', 'type', 'description'])
    df = df[df['type'].isin(['S', 'B'])]
    
    swung_descriptions = [
        'foul', 'hit_into_play', 'swinging_strike', 'foul_tip', 
        'swinging_strike_blocked', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip'
    ]
    df = df[~df['description'].isin(swung_descriptions)]
    
    df['is_strike'] = (df['type'] == 'S').astype(int)
    return df


def calculate_strike_probability(df, bins=25):
    x_bins = np.linspace(-1.5, 1.5, bins + 1)
    z_bins = np.linspace(0, 5, bins + 1)
    
    df = df.copy()
    df['x_bin'] = pd.cut(df['plate_x'], bins=x_bins, labels=False, include_lowest=True)
    df['z_bin'] = pd.cut(df['plate_z'], bins=z_bins, labels=False, include_lowest=True)
    
    df = df.dropna(subset=['x_bin', 'z_bin'])
    df['x_bin'] = df['x_bin'].astype(int)
    df['z_bin'] = df['z_bin'].astype(int)
    
    prob = df.groupby(['z_bin', 'x_bin']).agg(
        strike_prob=('is_strike', 'mean'),
        count=('is_strike', 'count')
    ).reset_index()
    
    prob_matrix = prob.pivot(index='z_bin', columns='x_bin', values='strike_prob')
    count_matrix = prob.pivot(index='z_bin', columns='x_bin', values='count')
    
    prob_matrix = prob_matrix.sort_index(ascending=False)
    count_matrix = count_matrix.sort_index(ascending=False)
    
    prob_matrix = prob_matrix.values
    count_matrix = count_matrix.values
    
    return prob_matrix, count_matrix, x_bins, z_bins


def extract_contour_paths(X, Z, prob):
    fig_t, ax_t = plt.subplots()
    paths = []
    try:
        cs = ax_t.contour(X, Z, prob, levels=[0.5])
        paths = [p.vertices for p in cs.get_paths() if len(p.vertices)]
    except Exception:
        pass
    plt.close(fig_t)
    return paths


def arc_length_weighted_mean_dist(paths, cx, cz):
    total_weight = 0.0
    total_weighted_dist = 0.0
    for verts in paths:
        if len(verts) < 2:
            d = np.hypot(verts[0, 0] - cx, verts[0, 1] - cz)
            total_weighted_dist += d
            total_weight += 1.0
            continue
        segs_start = verts[:-1]
        segs_end   = verts[1:]
        midpoints  = (segs_start + segs_end) / 2.0
        seg_lens   = np.hypot(segs_end[:, 0] - segs_start[:, 0],
                              segs_end[:, 1] - segs_start[:, 1])
        dists      = np.hypot(midpoints[:, 0] - cx,
                              midpoints[:, 1] - cz)
        total_weighted_dist += np.dot(seg_lens, dists)
        total_weight        += seg_lens.sum()

    return total_weighted_dist / total_weight if total_weight > 0 else np.nan


def compute_area_from_radius(radius_ft):
    radius_inches = radius_ft * 12
    return np.pi * radius_inches ** 2


def compute_grid_area_method_simple(df, threshold=0.5):
    """
    Simple grid method: divide into 1x1 inch squares, count squares with >=50% strikes.
    Each 1x1 inch square = 1 square foot = 144 square inches... wait, no.
    Plate is in FEET, so 1 inch = 1/12 feet.
    A 1 inch x 1 inch square = (1/12)^2 = 1/144 sq ft = 1 sq inch.
    """
    df = df.copy()
    
    grid_size = 1.0 / 12.0  # 1 inch in feet
    
    x_bins = np.arange(-1.5, 1.5 + grid_size, grid_size)
    z_bins = np.arange(0, 5 + grid_size, grid_size)
    
    df['x_bin'] = pd.cut(df['plate_x'], bins=x_bins, labels=False, include_lowest=True)
    df['z_bin'] = pd.cut(df['plate_z'], bins=z_bins, labels=False, include_lowest=True)
    
    df = df.dropna(subset=['x_bin', 'z_bin'])
    df['x_bin'] = df['x_bin'].astype(int)
    df['z_bin'] = df['z_bin'].astype(int)
    
    grid_stats = df.groupby(['x_bin', 'z_bin']).agg(
        strike_rate=('is_strike', 'mean'),
        count=('is_strike', 'count')
    ).reset_index()
    
    in_zone = (grid_stats['strike_rate'] >= threshold)
    area_sq_inches = in_zone.sum()
    
    return area_sq_inches


def compute_grid_area_method(prob_matrix, count_matrix, x_bins, z_bins, threshold=0.5, min_count=1):
    """Legacy method - now just calls simple method indirectly via df"""
    pass  # Not used anymore


def compute_all_areas(df, sz, contour_bins=30, min_count=20):
    results = []
    
    area_grid_all = compute_grid_area_method_simple(df, threshold=0.5)
    results.append(("All", len(df), area_grid_all, area_grid_all))
    
    for (balls, strikes) in counts:
        df_count = df[(df['balls'] == balls) & (df['strikes'] == strikes)]
        if len(df_count) < 50:
            continue
        
        area_from_grid = compute_grid_area_method_simple(df_count, threshold=0.5)
        
        results.append((f"{balls}-{strikes}", len(df_count), area_from_grid, area_from_grid))
    
    return results


def plot_sqrt_area_comparison_grid(area_results, sz, year, output_prefix=""):
    all_area = next(a for label, n, a, b in area_results if label == "All")
    all_sqrt = np.sqrt(all_area)

    lookup = {}
    for label, n, area_radius, area_grid in area_results:
        if label == "All":
            continue
        sqrt_area = np.sqrt(area_grid)
        lookup[label] = (area_grid, sqrt_area, sqrt_area - all_sqrt, n)

    balls_range = range(4)
    strikes_range = range(3)

    fig, axes = plt.subplots(3, 4, figsize=(11, 7), gridspec_kw=dict(hspace=0.15, wspace=0.12))

    cmap = mcolors.LinearSegmentedColormap.from_list("sz_div", ["#378ADD", "#f0f0f0", "#D85A30"])
    norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0.0, vmax=3)

    for s in strikes_range:
        for b in balls_range:
            ax = axes[s, b]
            ax.set_xticks([])
            ax.set_yticks([])

            key = f"{b}-{s}"

            if key not in lookup:
                ax.set_visible(False)
                continue

            avg_area, sqrt_area, vs_all_sqrt, n = lookup[key]
            color = cmap(norm(vs_all_sqrt))

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_edgecolor("#C04010" if vs_all_sqrt > 0.5 else "#1860A0" if vs_all_sqrt < -0.5 else "#aaaaaa")

            ax.set_facecolor(color)

            diff_color = "#7A2A08" if vs_all_sqrt > 0.5 else "#083A6E" if vs_all_sqrt < -0.5 else "#555555"

            ax.text(0.5, 0.82, key, transform=ax.transAxes, ha='center', va='center',
                    fontsize=13, fontweight='bold', color='#333333')

            ax.text(0.5, 0.57, f"Area = {avg_area:.1f} sq in", transform=ax.transAxes, ha='center', va='center',
                    fontsize=9.5, color='#222222')

            ax.text(0.5, 0.38, f"sqrt(Area) = {sqrt_area:.2f} in", transform=ax.transAxes, ha='center', va='center',
                    fontsize=9.5, color='#222222')

            ax.text(0.5, 0.14, f"{'+' if vs_all_sqrt >= 0 else ''}{vs_all_sqrt:.2f} in", transform=ax.transAxes, ha='center', va='center',
                    fontsize=11, fontweight='bold', color=diff_color)
            
            ax.text(0.02, 0.02, f"n={n:,}", transform=ax.transAxes, ha='left', va='bottom',
                    fontsize=7, color='#555555', alpha=0.8)

    for b in balls_range:
        axes[0, b].set_title(f"{b} ball{'s' if b != 1 else ''}", fontsize=11, pad=6)
    for s in strikes_range:
        axes[s, 0].set_ylabel(f"{s} strike{'s' if s != 1 else ''}", fontsize=11, rotation=90, labelpad=6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02, aspect=25)
    cbar.set_label('Delta sqrt(Area) vs All (in)', fontsize=11)

    fig.suptitle(
        f"Perceived Strike Zone Area by Count (Grid Method) - {year}\n"
        f"50% contour — baseline sqrt(Area) = {all_sqrt:.1f} in  (Area = {int(all_area)} sq in)",
        fontsize=12, y=1.01
    )

    filename = f"{output_prefix}sqrt_area_comparison_grid_{year}.png" if output_prefix else f"sqrt_area_comparison_grid_{year}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_grid_method_combined(df, sz, year, output_prefix="", contour_bins=30, min_count=20):
    grid_size = 1.0 / 12.0  # 1 inch in feet
    
    x_bins_grid = np.arange(-1.5, 1.5 + grid_size, grid_size)
    z_bins_grid = np.arange(0, 5 + grid_size, grid_size)
    
    df_all = df.copy()
    df_all['x_bin'] = pd.cut(df_all['plate_x'], bins=x_bins_grid, labels=False, include_lowest=True)
    df_all['z_bin'] = pd.cut(df_all['plate_z'], bins=z_bins_grid, labels=False, include_lowest=True)
    df_all = df_all.dropna(subset=['x_bin', 'z_bin'])
    df_all['x_bin'] = df_all['x_bin'].astype(int)
    df_all['z_bin'] = df_all['z_bin'].astype(int)
    
    grid_all = df_all.groupby(['x_bin', 'z_bin']).agg(
        strike_rate=('is_strike', 'mean'),
        count=('is_strike', 'count')
    ).reset_index()
    
    area_all = compute_grid_area_method_simple(df, threshold=0.5)
    n_total = len(df)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i, strikes in enumerate([0, 1, 2]):
        for j, balls in enumerate([0, 1, 2, 3]):
            ax = axes[i, j]
            
            df_count = df[(df['balls'] == balls) & (df['strikes'] == strikes)]
            if len(df_count) < 50:
                ax.set_visible(False)
                continue
            
            df_count = df_count.copy()
            df_count['x_bin'] = pd.cut(df_count['plate_x'], bins=x_bins_grid, labels=False, include_lowest=True)
            df_count['z_bin'] = pd.cut(df_count['plate_z'], bins=z_bins_grid, labels=False, include_lowest=True)
            df_count = df_count.dropna(subset=['x_bin', 'z_bin'])
            df_count['x_bin'] = df_count['x_bin'].astype(int)
            df_count['z_bin'] = df_count['z_bin'].astype(int)
            
            grid_count = df_count.groupby(['x_bin', 'z_bin']).agg(
                strike_rate=('is_strike', 'mean'),
                count=('is_strike', 'count')
            ).reset_index()
            
            area_from_grid = len(grid_count[grid_count['strike_rate'] >= 0.5])
            diff_val = area_from_grid - area_all
            
            ax.set_xlim(-1, 1)
            ax.set_ylim(1.25, 3.75)
            ax.set_aspect('equal')
            
            ax.axhline(1.5, color='gray', linestyle='-', alpha=0.8, linewidth=2.5)
            ax.axhline(3.5, color='gray', linestyle='-', alpha=0.8, linewidth=2.5)
            ax.axvline(-0.83, color='gray', linestyle='-', alpha=0.8, linewidth=2.5)
            ax.axvline(0.83, color='gray', linestyle='-', alpha=0.8, linewidth=2.5)
            
            grid_all['in_zone'] = grid_all['strike_rate'] >= 0.5
            grid_count['in_zone'] = grid_count['strike_rate'] >= 0.5
            
            n_x_bins = len(x_bins_grid) - 1
            n_z_bins = len(z_bins_grid) - 1
            
            all_in_zone = np.zeros((n_z_bins, n_x_bins))
            for _, row in grid_all.iterrows():
                if 0 <= row['x_bin'] < n_x_bins and 0 <= row['z_bin'] < n_z_bins:
                    all_in_zone[int(row['z_bin']), int(row['x_bin'])] = 1 if row['in_zone'] else 0
            
            count_in_zone = np.zeros((n_z_bins, n_x_bins))
            for _, row in grid_count.iterrows():
                if 0 <= row['x_bin'] < n_x_bins and 0 <= row['z_bin'] < n_z_bins:
                    count_in_zone[int(row['z_bin']), int(row['x_bin'])] = 1 if row['in_zone'] else 0
            
            red_green_cmap = ListedColormap(['#ff4444', '#44ff44'])
            
            ax.pcolormesh(x_bins_grid, z_bins_grid, count_in_zone, 
                         cmap=red_green_cmap, shading='flat', alpha=0.7, edgecolor='white', linewidth=0.1)
            
            ax.text(
                0.5, 1.1, f"{balls}-{strikes}",
                transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=16, fontweight='bold'
            )
            
            ax.text(
                0.5, 1.11,
                f"Area: {int(area_from_grid)} sq in (n={len(df_count):,})\n(All Pitches: {int(area_all)} sq in, Diff: {diff_val:+.0f} sq in)",
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=10
            )
            
            if j == 0:
                ax.set_ylabel(f'{strikes} strike', fontsize=10)
            if i == 2:
                ax.set_xlabel(f'{balls} ball', fontsize=10)
    
    plt.suptitle(f'Grid Method Visualization - {year} (n={n_total:,})\nGreen = >=50% strikes in 1x1in square, Gray = Approx Zone', fontsize=15, y=1.01)
    plt.tight_layout()
    
    filename = f"{output_prefix}grid_method_combined_{year}.png" if output_prefix else f"grid_method_combined_{year}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_area_vs_year_by_count(years_data, years_to_plot=None, output_suffix=""):
    from scipy import stats
    
    if years_to_plot is None:
        years_to_plot = sorted(years_data.keys())
    
    fig = plt.figure(figsize=(14, 14))
    
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.25)
    
    all_counts_areas = []
    for (balls, strikes) in counts:
        key = f"{balls}-{strikes}"
        for year in years_to_plot:
            area = years_data[year].get(key)
            if area is not None:
                all_counts_areas.append(area)
    
    global_min = min(all_counts_areas)
    global_max = max(all_counts_areas)
    y_margin = (global_max - global_min) * 0.1
    global_ylim = (global_min - y_margin, global_max + y_margin)
    
    ax_all = fig.add_subplot(gs[0, 1:3])
    
    areas_all = []
    for year in years_to_plot:
        area = years_data[year].get("All")
        if area is not None:
            areas_all.append((year, area))
    
    years_list_all = [a[0] for a in areas_all]
    area_vals_all = [a[1] for a in areas_all]
    
    ax_all.plot(years_list_all, area_vals_all, 'o-', linewidth=2.5, markersize=8, color='#D85A30')
    ax_all.fill_between(years_list_all, area_vals_all, alpha=0.2, color='#D85A30')
    
    if len(years_list_all) >= 2:
        slope, intercept, r, p, se = stats.linregress(years_list_all, area_vals_all)
        x_fit = np.array([min(years_list_all), max(years_list_all)])
        y_fit = slope * x_fit + intercept
        ax_all.plot(x_fit, y_fit, '--', color='gray', alpha=0.6, linewidth=2, label=f'trend: {slope:+.1f}/yr')
        ax_all.legend(fontsize=8, loc='best')
    
    ax_all.set_ylim(global_ylim)
    ax_all.set_title('All Pitches', fontsize=12, fontweight='bold')
    ax_all.set_ylabel('Area (sq in)', fontsize=10)
    ax_all.grid(True, alpha=0.3)
    ax_all.set_xticks(years_to_plot)
    ax_all.tick_params(axis='x', rotation=45)
    ax_all.set_xlabel('Year', fontsize=10)
    
    for i, strikes in enumerate([0, 1, 2]):
        for j, balls in enumerate([0, 1, 2, 3]):
            ax = fig.add_subplot(gs[i+1, j])
            key = f"{balls}-{strikes}"
            
            areas = []
            for year in years_to_plot:
                area = years_data[year].get(key)
                if area is not None:
                    areas.append((year, area))
            
            if not areas:
                ax.set_visible(False)
                continue
            
            years_list = [a[0] for a in areas]
            area_vals = [a[1] for a in areas]
            
            ax.plot(years_list, area_vals, 'o-', linewidth=2, markersize=6, color='#1860A0')
            ax.fill_between(years_list, area_vals, alpha=0.2, color='#1860A0')
            
            if len(years_list) >= 2:
                slope, intercept, r, p, se = stats.linregress(years_list, area_vals)
                x_fit = np.array([min(years_list), max(years_list)])
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.6, linewidth=1.5, label=f'trend: {slope:+.1f}/yr')
                ax.legend(fontsize=7, loc='best')
            
            ax.set_ylim(global_ylim)
            
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel('Area (sq in)', fontsize=10)
            ax.set_title(f'{key}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(years_to_plot)
            ax.tick_params(axis='x', rotation=45)
    
    filename = f"area_vs_year_by_count{output_suffix}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_sqrt_area_vs_year_by_count(years_data, years_to_plot=None, output_suffix=""):
    from scipy import stats
    
    if years_to_plot is None:
        years_to_plot = sorted(years_data.keys())
    
    fig = plt.figure(figsize=(14, 14))
    
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.25)
    
    all_counts_sqrt = []
    for (balls, strikes) in counts:
        key = f"{balls}-{strikes}"
        for year in years_to_plot:
            area = years_data[year].get(key)
            if area is not None:
                all_counts_sqrt.append(np.sqrt(area))
    
    global_min = min(all_counts_sqrt)
    global_max = max(all_counts_sqrt)
    y_margin = (global_max - global_min) * 0.1
    global_ylim = (global_min - y_margin, global_max + y_margin)
    
    ax_all = fig.add_subplot(gs[0, 1:3])
    
    sqrt_areas_all = []
    for year in years_to_plot:
        area = years_data[year].get("All")
        if area is not None:
            sqrt_areas_all.append((year, np.sqrt(area)))
    
    years_list_all = [a[0] for a in sqrt_areas_all]
    sqrt_vals_all = [a[1] for a in sqrt_areas_all]
    
    ax_all.plot(years_list_all, sqrt_vals_all, 'o-', linewidth=2.5, markersize=8, color='#D85A30')
    ax_all.fill_between(years_list_all, sqrt_vals_all, alpha=0.2, color='#D85A30')
    
    if len(years_list_all) >= 2:
        slope, intercept, r, p, se = stats.linregress(years_list_all, sqrt_vals_all)
        x_fit = np.array([min(years_list_all), max(years_list_all)])
        y_fit = slope * x_fit + intercept
        ax_all.plot(x_fit, y_fit, '--', color='gray', alpha=0.6, linewidth=2, label=f'trend: {slope:+.3f}/yr')
        ax_all.legend(fontsize=8, loc='best')
    
    ax_all.set_ylim(global_ylim)
    ax_all.set_title('All Pitches', fontsize=12, fontweight='bold')
    ax_all.set_ylabel('sqrt(Area) (in)', fontsize=10)
    ax_all.grid(True, alpha=0.3)
    ax_all.set_xticks(years_to_plot)
    ax_all.tick_params(axis='x', rotation=45)
    ax_all.set_xlabel('Year', fontsize=10)
    
    for i, strikes in enumerate([0, 1, 2]):
        for j, balls in enumerate([0, 1, 2, 3]):
            ax = fig.add_subplot(gs[i+1, j])
            key = f"{balls}-{strikes}"
            
            sqrt_areas = []
            for year in years_to_plot:
                area = years_data[year].get(key)
                if area is not None:
                    sqrt_areas.append((year, np.sqrt(area)))
            
            if not sqrt_areas:
                ax.set_visible(False)
                continue
            
            years_list = [a[0] for a in sqrt_areas]
            sqrt_vals = [a[1] for a in sqrt_areas]
            
            ax.plot(years_list, sqrt_vals, 'o-', linewidth=2, markersize=6, color='#1860A0')
            ax.fill_between(years_list, sqrt_vals, alpha=0.2, color='#1860A0')
            
            if len(years_list) >= 2:
                slope, intercept, r, p, se = stats.linregress(years_list, sqrt_vals)
                x_fit = np.array([min(years_list), max(years_list)])
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.6, linewidth=1.5, label=f'trend: {slope:+.3f}/yr')
                ax.legend(fontsize=7, loc='best')
            
            ax.set_ylim(global_ylim)
            
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel('sqrt(Area) (in)', fontsize=10)
            ax.set_title(f'{key}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(years_to_plot)
            ax.tick_params(axis='x', rotation=45)
    
    filename = f"sqrt_area_vs_year_by_count{output_suffix}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_area_vs_year_all_pitches(years_data, years_to_plot=None, output_suffix=""):
    from scipy import stats
    
    if years_to_plot is None:
        years_to_plot = sorted(years_data.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    areas = []
    for year in years_to_plot:
        area = years_data[year].get("All")
        if area is not None:
            areas.append((year, area))
    
    years_list = [a[0] for a in areas]
    area_vals = [a[1] for a in areas]
    
    ax.plot(years_list, area_vals, 'o-', linewidth=2.5, markersize=8, color='#D85A30')
    ax.fill_between(years_list, area_vals, alpha=0.2, color='#D85A30')
    
    if len(years_list) >= 2:
        slope, intercept, r, p, se = stats.linregress(years_list, area_vals)
        x_fit = np.array([min(years_list), max(years_list)])
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.6, linewidth=2, label=f'trend: {slope:+.1f}/yr')
        ax.legend(fontsize=10, loc='best')
    
    y_margin = (max(area_vals) - min(area_vals)) * 0.2
    ax.set_ylim(min(area_vals) - y_margin, max(area_vals) + y_margin)
    
    for year, area in areas:
        ax.annotate(f'{int(area)}', (year, area), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Area (sq in)', fontsize=12)
    ax.set_title('Overall Strike Zone Area Over Time (Grid Method)\n50% contour - All Called Pitches', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years_to_plot)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    filename = f"area_vs_year_all_pitches{output_suffix}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_sqrt_area_vs_year_all_pitches(years_data, years_to_plot=None, output_suffix=""):
    from scipy import stats
    
    if years_to_plot is None:
        years_to_plot = sorted(years_data.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sqrt_areas = []
    for year in years_to_plot:
        area = years_data[year].get("All")
        if area is not None:
            sqrt_areas.append((year, np.sqrt(area)))
    
    years_list = [a[0] for a in sqrt_areas]
    sqrt_vals = [a[1] for a in sqrt_areas]
    
    ax.plot(years_list, sqrt_vals, 'o-', linewidth=2.5, markersize=8, color='#D85A30')
    ax.fill_between(years_list, sqrt_vals, alpha=0.2, color='#D85A30')
    
    if len(years_list) >= 2:
        slope, intercept, r, p, se = stats.linregress(years_list, sqrt_vals)
        x_fit = np.array([min(years_list), max(years_list)])
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.6, linewidth=2, label=f'trend: {slope:+.3f}/yr')
        ax.legend(fontsize=10, loc='best')
    
    y_margin = (max(sqrt_vals) - min(sqrt_vals)) * 0.2
    ax.set_ylim(min(sqrt_vals) - y_margin, max(sqrt_vals) + y_margin)
    
    for year, sqrt_area in sqrt_areas:
        ax.annotate(f'{sqrt_area:.1f}', (year, sqrt_area), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('sqrt(Area) (in)', fontsize=12)
    ax.set_title('Overall Strike Zone sqrt(Area) Over Time (Grid Method)\n50% contour - All Called Pitches', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years_to_plot)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    filename = f"sqrt_area_vs_year_all_pitches{output_suffix}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def run_temporal_analysis(years=range(2018, 2026)):
    print("=" * 70)
    print("TEMPORAL STRIKE ZONE ANALYSIS")
    print("=" * 70)
    
    years_data = {}
    all_years_area_results = {}
    
    for year in years:
        print(f"\n--- Processing {year} ---")
        df = load_data_for_year(year)
        print(f"Loaded {len(df):,} called pitches for {year}")
        
        area_results = compute_all_areas(df, THEORETICAL_STRIKE_ZONE, contour_bins=30, min_count=20)
        years_data[year] = {label: area_grid for label, n, area_radius, area_grid in area_results}
        all_years_area_results[year] = area_results
        
        print(f"\n  Computing plots for {year}...")
        plot_sqrt_area_comparison_grid(area_results, THEORETICAL_STRIKE_ZONE, year)
        plot_grid_method_combined(df, THEORETICAL_STRIKE_ZONE, year)
        
        print(f"\n  Strike Zone Areas for {year}:")
        for label, n, area_radius, area_grid in area_results:
            print(f"    {label}: {area_grid:.1f} sq in (n={n:,})")
    
    print("\n" + "=" * 70)
    print("GENERATING TEMPORAL TREND PLOTS")
    print("=" * 70)
    
    years_for_plotting = [y for y in sorted(years_data.keys()) if y != 2020]
    
    plot_area_vs_year_by_count(years_data, years_to_plot=years_for_plotting)
    plot_sqrt_area_vs_year_by_count(years_data, years_to_plot=years_for_plotting)
    plot_area_vs_year_all_pitches(years_data, years_to_plot=years_for_plotting)
    plot_sqrt_area_vs_year_all_pitches(years_data, years_to_plot=years_for_plotting)
    
    print("\n" + "=" * 70)
    print("ALL PLOTS SAVED TO:", PLOTS_DIR)
    print("=" * 70)


if __name__ == "__main__":
    run_temporal_analysis(years=range(2018, 2026))
