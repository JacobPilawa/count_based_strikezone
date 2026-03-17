import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import os

DATA_DIR = "../data"
PLOTS_DIR = "../plots"

THEORETICAL_STRIKE_ZONE = {
    'plate_x_min': -0.83,
    'plate_x_max': 0.83,
    'plate_z_min': 1.5,
    'plate_z_max': 3.5
}

def load_data():
    dfs = []
    for year in [2023, 2024, 2025]:
        path = os.path.join(DATA_DIR, f"statcast_{year}.parquet")
        df = pd.read_parquet(path)
        df['year'] = year
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=['plate_x', 'plate_z', 'type', 'description'])
    combined = combined[combined['type'].isin(['S', 'B'])]
    
    swung_descriptions = [
        'foul', 'hit_into_play', 'swinging_strike', 'foul_tip', 
        'swinging_strike_blocked', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip'
    ]
    combined = combined[~combined['description'].isin(swung_descriptions)]
    
    combined['is_strike'] = (combined['type'] == 'S').astype(int)
    return combined

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

def create_all_contours_plot(df, x_bins, z_bins, min_count=20):
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
    
    colors = plt.cm.RdYlBu_r(np.array([hitter_to_pitcher[c] for c in counts]))
    
    fig, ax = plt.subplots(figsize=(14, 11))
    
    df_all = df.copy()
    prob_all, count_all, _, _ = calculate_strike_probability(df_all, bins=25)
    prob_all_filtered = np.where(count_all >= min_count, prob_all, np.nan)
    
    im = ax.imshow(
        prob_all_filtered,
        extent=[x_bins[0], x_bins[-1], z_bins[0], z_bins[-1]],
        origin='lower',
        cmap='bwr',
        vmin=0,
        vmax=1,
        aspect='auto',
        alpha=0.2
    )
    
    ax.add_patch(Rectangle(
        (THEORETICAL_STRIKE_ZONE['plate_x_min'], THEORETICAL_STRIKE_ZONE['plate_z_min']),
        THEORETICAL_STRIKE_ZONE['plate_x_max'] - THEORETICAL_STRIKE_ZONE['plate_x_min'],
        THEORETICAL_STRIKE_ZONE['plate_z_max'] - THEORETICAL_STRIKE_ZONE['plate_z_min'],
        linewidth=3,
        edgecolor='lime',
        facecolor='none',
        label='Theoretical Zone'
    ))
    
    x_centers = x_bins[:-1] + (x_bins[1]-x_bins[0])/2
    z_centers = z_bins[:-1] + (z_bins[1]-z_bins[0])/2
    X, Z = np.meshgrid(x_centers, z_centers)
    
    legend_elements = [
        Rectangle((0,0), 1, 1, facecolor='none', edgecolor='lime', linewidth=3, label='Theoretical Zone'),
        Line2D([0], [0], color='black', linewidth=3, label='All Pitches (50%)'),
    ]
    
    contour_data = {}
    for idx, (balls, strikes) in enumerate(counts):
        df_count = df[(df['balls'] == balls) & (df['strikes'] == strikes)]
        
        if len(df_count) < 50:
            continue
        
        prob_matrix, count_matrix, _, _ = calculate_strike_probability(df_count, bins=25)
        prob_filtered = np.where(count_matrix >= min_count, prob_matrix, np.nan)
        
        contour_data[idx] = (prob_filtered, colors[idx])
        
        try:
            cs = ax.contour(
                X, Z, prob_filtered,
                levels=[0.5],
                colors=[colors[idx]],
                linewidths=2.5,
                linestyles='dashed'
            )
        except:
            pass
    
    try:
        cs_all = ax.contour(
            X, Z, prob_all_filtered,
            levels=[0.5],
            colors=['black'],
            linewidths=3,
            linestyles='solid'
        )
    except:
        pass
    
    for idx, (balls, strikes) in enumerate(counts):
        df_count = df[(df['balls'] == balls) & (df['strikes'] == strikes)]
        if len(df_count) >= 50:
            legend_elements.append(
                Line2D([0], [0], color=colors[idx], linestyle='dashed', linewidth=2.5, label=f'{balls}-{strikes}')
            )
    
    ax.set_xlim(-2.2, 1.5)
    ax.set_ylim(-0.5, 5.5)
    
    ax.set_xlabel('Plate X (feet)', fontsize=12)
    ax.set_ylabel('Plate Z (feet)', fontsize=12)
    ax.set_title("MLB 2023-2025: 50% Strike Probability Contours by Count\n(Pitchers counts = blue, Hitters counts = red)", fontsize=14, fontweight='bold')
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(2.5, color='gray', linestyle='--', alpha=0.5)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Strike Probability (All Pitches)', fontsize=10)
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=2)
    
    axins_top = ax.inset_axes([0.08, 0.73, 0.30, 0.22])
    for idx, data in contour_data.items():
        prob_filtered, color = data
        axins_top.contour(X, Z, prob_filtered, levels=[0.5], colors=[color], linewidths=1.2, linestyles='dashed')
    axins_top.set_xlim(-0.75, 0.75)
    axins_top.set_ylim(3.3, 3.6)
    axins_top.set_title('Top Edge', fontsize=8, fontweight='bold')
    axins_top.tick_params(labelsize=6)
    axins_top.axhline(3.5, color='lime', linewidth=1, linestyle='--', alpha=0.7)
    for spine in axins_top.spines.values():
        spine.set_linewidth(1.5)
    
    axins_bottom = ax.inset_axes([0.08, 0.05, 0.30, 0.22])
    for idx, data in contour_data.items():
        prob_filtered, color = data
        axins_bottom.contour(X, Z, prob_filtered, levels=[0.5], colors=[color], linewidths=1.2, linestyles='dashed')
    axins_bottom.set_xlim(-0.75, 0.75)
    axins_bottom.set_ylim(1.4, 1.7)
    axins_bottom.set_title('Bottom Edge', fontsize=8, fontweight='bold')
    axins_bottom.tick_params(labelsize=6)
    axins_bottom.axhline(1.5, color='lime', linewidth=1, linestyle='--', alpha=0.7)
    for spine in axins_bottom.spines.values():
        spine.set_linewidth(1.5)
    
    axins_left = ax.inset_axes([-0.30, 0.32, 0.24, 0.36])
    for idx, data in contour_data.items():
        prob_filtered, color = data
        axins_left.contour(X, Z, prob_filtered, levels=[0.5], colors=[color], linewidths=1.2, linestyles='dashed')
    axins_left.set_xlim(-1.0, -0.7)
    axins_left.set_ylim(1.25, 3.75)
    axins_left.set_title('Left Edge', fontsize=8, fontweight='bold')
    axins_left.tick_params(labelsize=6)
    axins_left.axvline(-0.83, color='lime', linewidth=1, linestyle='--', alpha=0.7)
    for spine in axins_left.spines.values():
        spine.set_linewidth(1.5)
    
    axins_right = ax.inset_axes([1.06, 0.32, 0.24, 0.36])
    for idx, data in contour_data.items():
        prob_filtered, color = data
        axins_right.contour(X, Z, prob_filtered, levels=[0.5], colors=[color], linewidths=1.2, linestyles='dashed')
    axins_right.set_xlim(0.8, 1.0)
    axins_right.set_ylim(1.25, 3.75)
    axins_right.set_title('Right Edge', fontsize=8, fontweight='bold')
    axins_right.tick_params(labelsize=6)
    axins_right.axvline(0.83, color='lime', linewidth=1, linestyle='--', alpha=0.7)
    for spine in axins_right.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "strike_probability_all_contours.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: strike_probability_all_contours.png")

if __name__ == "__main__":
    print("Loading data (2023-2025)...")
    df = load_data()
    print(f"Loaded {len(df):,} called pitches total\n")
    
    x_bins = np.linspace(-1.5, 1.5, 26)
    z_bins = np.linspace(0, 5, 26)
    
    print("Creating all contours plot with insets...")
    create_all_contours_plot(df, x_bins, z_bins)
    
    print("\nDone!")
