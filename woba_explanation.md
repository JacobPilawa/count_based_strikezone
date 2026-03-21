# wOBA Location Analysis (`woba_analysis_codex.py`)

## Data scope
- Source: `data/statcast_2023.parquet`, `statcast_2024.parquet`, `statcast_2025.parquet`
- Seasons: 2023–2025 combined
- Unit of analysis: pitch location (`plate_x`, `plate_z`)
- Rows with missing location are dropped.

## Strike-zone framing and binning
- The plot framing matches the strike-zone notebook setup:
  - `plate_x` range: `-1.5` to `1.5` feet
  - `plate_z` range: `0.0` to `5.0` feet
- Bins are defined two ways:
  1. **Fine grid:** 1-inch × 1-inch cells across the plotting window (36 `x` bins × 60 `z` bins). These match the original strike-zone analysis and are saved at `plots/woba_analysis_codex/`.
  2. **Coarse grid:** the strike zone is divided into 4×4 equal cells (`~0.415 ft` wide × `0.5 ft` tall) and that spacing is continued outward until the entire `X_RANGE`/`Z_RANGE` is covered. These larger bins are saved under `plots/woba_analysis_codex/coarse4x4/`.
- The theoretical strike zone drawn on each plot is:
  - `plate_x`: `[-0.83, 0.83]`
  - `plate_z`: `[1.5, 3.5]`

## Metric definitions by location cell
No minimum-pitch filter is applied. Any cell with available data is colored.

### 1. wOBA
For each bin:
- Numerator: `sum(woba_value)`
- Denominator: `sum(woba_denom)`
- Formula: `wOBA = sum(woba_value) / sum(woba_denom)`

Only pitches with `woba_denom == 1` and non-null `woba_value` appear in this aggregation, i.e., pitches where Statcast defines a wOBA outcome.

### 2. Batting Average (BA)
For each bin, using at-bat-ending events:
- `AB_EVENTS` = {
  `single`, `double`, `triple`, `home_run`,
  `strikeout`, `strikeout_double_play`,
  `field_out`, `force_out`, `double_play`, `grounded_into_double_play`,
  `fielders_choice`, `fielders_choice_out`, `triple_play`
}
- `HIT_EVENTS` = {`single`, `double`, `triple`, `home_run`}
- Formula: `BA = hits / AB`

### 3. Slugging (SLG)
For the same AB bins:
- Total bases mapping:
  - `single=1`, `double=2`, `triple=3`, `home_run=4`
  - outs/non-hit AB events = `0`
- Formula: `SLG = total_bases / AB`

### 4. Pitch count
For each bin:
- Count all pitches with valid (`plate_x`, `plate_z`) in that bin.
- Plot includes a numeric label in each non-empty cell.

## Combined multi-panel grids
- `plots/woba_analysis_codex/woba_grid_all_types.png` shows `All pitches` plus every pitch type in one figure for wOBA; equivalent aligned figures exist for BA (`ba_grid_all_types.png`) and SLG (`slg_grid_all_types.png`).
- Each grid set shares global color limits: wOBA is clipped to the 0.15–0.45 window, while BA/SLG use the empirical 5th/95th percentile bounds computed across all pitch types.

## Coarse-grid multi-panel figures
- The folder `plots/woba_analysis_codex/coarse4x4/` contains versions of the same multi-panel grids but using the larger 4×4 strike-zone bins: `woba_grid_all_types_coarse4x4.png`, `ba_grid_all_types_coarse4x4.png`, and `slg_grid_all_types_coarse4x4.png`. These figures also annotate each coarse cell with its value, making the coarser buckets easy to read.

## Per-pitch-type outputs
- For each configuration you get a directory tree:
  - Fine bins: `plots/woba_analysis_codex/pitch_types_fine/<pitch_type>/...`
  - Coarse bins: `plots/woba_analysis_codex/coarse4x4/pitch_types_coarse/<pitch_type>/...`
- Inside each `<pitch_type>` folder are four plots named:
  - `woba_heatmap_2023-25_<config>_<pitch_type>.png`
  - `ba_heatmap_2023-25_<config>_<pitch_type>.png`
  - `slg_heatmap_2023-25_<config>_<pitch_type>.png`
  - `pitch_count_heatmap_2023-25_<config>_<pitch_type>.png`
  where `<config>` is either `fine` or `coarse4x4`.
- The same naming applies for the `all` key inside each config root (e.g., `woba_heatmap_2023-25_fine_all.png` and `woba_heatmap_2023-25_coarse4x4_all.png`).

## Output files
Main output directory hierarchies now look like:
- `plots/woba_analysis_codex/` (fine, 1-inch bins)
  - Heatmaps: `*_woba_heatmap_2023-25_fine_all.png`, `*_ba_heatmap_2023-25_fine_all.png`, etc.
  - Multi-panel grids: `*_grid_all_types.png` for each metric.
  - Per-pitch-type subfolders: `pitch_types_fine/<pitch_type>/...`
- `plots/woba_analysis_codex/coarse4x4/` (coarse bins)
  - Heatmaps and grids follow the same pattern but include `coarse4x4` in the file names and path.
  - Per-pitch-type subfolders: `pitch_types_coarse/<pitch_type>/...`
