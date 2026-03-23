#!/usr/bin/env python3
"""
Generate a static HTML dashboard showing dataset inventory by pattern type.

Reads from:
  - data/labels/patients.csv
  - data/labels/segments.csv
  - data/labels/harvest_manifest.json
  - data/labels/bipd_harvest_manifest.json  (optional)
  - data/labels/rda_harvest_manifest.json   (optional)
  - data/labels/discharge_times.json
  - data/labels/channel_involvement.json
  - data/labels/annotations.csv

Outputs:
  - results/dataset_dashboard.html

Re-run to update:
  conda run -n foe python code/generate_dataset_dashboard.py

Auto-update every 30 seconds:
  while true; do conda run -n foe python code/generate_dataset_dashboard.py; sleep 30; done
"""

import json
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "labels"
OUT = ROOT / "results" / "dataset_dashboard.html"


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    # ── Load data sources ────────────────────────────────────────────
    patients = pd.read_csv(LABELS / "patients.csv")
    segments = pd.read_csv(LABELS / "segments.csv")
    harvest = load_json(LABELS / "harvest_manifest.json")
    bipd_harvest = load_json(LABELS / "bipd_harvest_manifest.json")
    rda_harvest = load_json(LABELS / "rda_harvest_manifest.json")
    discharge_times = load_json(LABELS / "discharge_times.json")
    channel_inv = load_json(LABELS / "channel_involvement.json")
    annotations = pd.read_csv(LABELS / "annotations.csv") if (LABELS / "annotations.csv").exists() else pd.DataFrame()

    # ── Existing counts from patients.csv ────────────────────────────
    subtype_counts = patients["subtype"].value_counts().to_dict()
    existing = {
        "lpd": subtype_counts.get("lpd", 0),
        "gpd": subtype_counts.get("gpd", 0),
        "lrda": subtype_counts.get("lrda", 0),
        "grda": subtype_counts.get("grda", 0),
    }

    # Segment counts by subtype from segments.csv
    seg_counts = segments["subtype"].value_counts().to_dict()
    existing_segs = {
        "lpd": seg_counts.get("lpd", 0),
        "gpd": seg_counts.get("gpd", 0),
        "lrda": seg_counts.get("lrda", 0),
        "grda": seg_counts.get("grda", 0),
    }

    # ── Harvest manifest (LPD + seizure) ─────────────────────────────
    harvest_lpd = 0
    harvest_highfreq = 0
    for pid, info in harvest.items():
        est_freq = info.get("est_freq", 0)
        if est_freq and est_freq >= 2.5:
            harvest_highfreq += 1
        else:
            harvest_lpd += 1

    # ── BIPD harvest ─────────────────────────────────────────────────
    bipd_count = len(bipd_harvest)

    # ── Other harvest ──────────────────────────────────────────────
    other_harvest = load_json(LABELS / "other_harvest_manifest.json")
    other_count = len(other_harvest)

    # ── RDA harvest ──────────────────────────────────────────────────
    rda_lrda = 0
    rda_grda = 0
    for pid, info in rda_harvest.items():
        st = info.get("subtype", "").lower()
        if st == "lrda":
            rda_lrda += 1
        elif st == "grda":
            rda_grda += 1
        else:
            # default: count as unknown
            rda_lrda += 1  # or skip

    # ── Label status ─────────────────────────────────────────────────
    # Frequency labels
    freq_by_type = {}
    for st in ["lpd", "gpd", "lrda", "grda"]:
        mask = (patients["subtype"] == st) & patients["gold_standard_freq"].notna()
        freq_by_type[st] = int(mask.sum())

    # Laterality labels (non-empty string)
    lat_col = patients["laterality"].fillna("")
    lat_by_type = {}
    for st in ["lpd", "gpd", "lrda", "grda"]:
        mask = (patients["subtype"] == st) & (lat_col != "")
        lat_by_type[st] = int(mask.sum())

    # Discharge timing labels
    dt_patients = set(discharge_times.keys()) if isinstance(discharge_times, dict) else set()
    dt_by_type = {}
    for st in ["lpd", "gpd", "lrda", "grda"]:
        pids_st = set(patients[patients["subtype"] == st]["patient_id"].astype(str))
        dt_by_type[st] = len(pids_st & dt_patients)

    # Spatial labels (channel involvement)
    ci_patients = set(channel_inv.keys()) if isinstance(channel_inv, dict) else set()
    ci_by_type = {}
    for st in ["lpd", "gpd", "lrda", "grda"]:
        pids_st = set(patients[patients["subtype"] == st]["patient_id"].astype(str))
        ci_by_type[st] = len(pids_st & ci_patients)

    # ── Frequency distribution ───────────────────────────────────────
    freqs = patients["gold_standard_freq"].dropna().tolist()
    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 100]
    bin_labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", "3+"]
    freq_hist = [0] * len(bin_labels)
    for f in freqs:
        for i in range(len(bins) - 1):
            if bins[i] <= f < bins[i + 1]:
                freq_hist[i] += 1
                break

    # ── Totals ───────────────────────────────────────────────────────
    total_existing_patients = sum(existing.values())
    total_existing_segments = sum(existing_segs.values())
    total_harvested = harvest_lpd + harvest_highfreq + bipd_count + rda_lrda + rda_grda + other_count
    total_all = total_existing_segments + total_harvested

    # ── Bar chart data ───────────────────────────────────────────────
    bar_categories = [
        "LPD\n(existing)",
        "LPD\n(harvested)",
        "GPD\n(existing)",
        "LRDA\n(existing)",
        "LRDA\n(harvested)",
        "GRDA\n(existing)",
        "GRDA\n(harvested)",
        "BIPD\n(harvested)",
        "Hi-freq LPD\n(>2.5 Hz)",
        "Other\n(controls)",
    ]
    bar_values = [
        existing_segs["lpd"],
        harvest_lpd,
        existing_segs["gpd"],
        existing_segs["lrda"],
        rda_lrda,
        existing_segs["grda"],
        rda_grda,
        bipd_count,
        harvest_highfreq,
        other_count,
    ]
    bar_colors = [
        "#2166ac",  # LPD existing (dark blue)
        "#92c5de",  # LPD harvested (light blue)
        "#b2182b",  # GPD existing (dark red)
        "#1b7837",  # LRDA existing (dark green)
        "#a6dba0",  # LRDA harvested (light green)
        "#762a83",  # GRDA existing (dark purple)
        "#c2a5cf",  # GRDA harvested (light purple)
        "#e08214",  # BIPD (orange)
        "#fee090",  # high-freq LPD (yellow)
        "#888888",  # Other controls (gray)
    ]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Generate HTML ────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="10">
<title>Dataset Inventory Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; padding: 20px; }}
  h1 {{ text-align: center; font-size: 28px; margin-bottom: 6px; color: #fff; }}
  .subtitle {{ text-align: center; color: #888; font-size: 13px; margin-bottom: 20px; }}
  .summary-bar {{ display: flex; justify-content: center; gap: 30px; margin-bottom: 28px; }}
  .summary-card {{ background: #1a1d27; border: 1px solid #333; border-radius: 10px; padding: 16px 28px; text-align: center; min-width: 150px; }}
  .summary-card .number {{ font-size: 36px; font-weight: 700; color: #4fc3f7; }}
  .summary-card .label {{ font-size: 12px; color: #999; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}
  .section {{ background: #1a1d27; border: 1px solid #333; border-radius: 10px; padding: 24px; margin-bottom: 20px; }}
  .section h2 {{ font-size: 18px; margin-bottom: 16px; color: #ccc; border-bottom: 1px solid #333; padding-bottom: 8px; }}
  /* Bar chart */
  .bar-chart {{ display: flex; align-items: flex-end; gap: 8px; height: 260px; padding: 0 10px; }}
  .bar-col {{ display: flex; flex-direction: column; align-items: center; flex: 1; height: 100%; justify-content: flex-end; }}
  .bar {{ border-radius: 4px 4px 0 0; min-width: 40px; width: 100%; transition: height 0.3s; position: relative; }}
  .bar-val {{ font-size: 13px; font-weight: 600; color: #fff; margin-bottom: 4px; }}
  .bar-label {{ font-size: 11px; color: #999; margin-top: 8px; text-align: center; white-space: pre-line; line-height: 1.3; }}
  /* Table */
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ text-align: left; padding: 10px 12px; background: #252830; color: #aaa; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
  td {{ padding: 10px 12px; border-top: 1px solid #2a2d37; }}
  tr:hover td {{ background: #22252f; }}
  .type-badge {{ display: inline-block; padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 12px; }}
  /* Freq histogram */
  .freq-chart {{ display: flex; align-items: flex-end; gap: 6px; height: 160px; padding: 0 10px; }}
  .freq-col {{ display: flex; flex-direction: column; align-items: center; flex: 1; height: 100%; justify-content: flex-end; }}
  .freq-bar {{ background: #4fc3f7; border-radius: 3px 3px 0 0; min-width: 30px; width: 100%; }}
  .freq-val {{ font-size: 12px; color: #aaa; margin-bottom: 3px; }}
  .freq-label {{ font-size: 11px; color: #888; margin-top: 6px; }}
  /* Harvest status */
  .harvest-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
  .harvest-card {{ background: #252830; border-radius: 8px; padding: 16px; }}
  .harvest-card h3 {{ font-size: 14px; color: #bbb; margin-bottom: 8px; }}
  .harvest-stat {{ font-size: 28px; font-weight: 700; color: #4fc3f7; }}
  .harvest-detail {{ font-size: 12px; color: #888; margin-top: 4px; }}
  .zero {{ color: #555; }}
</style>
</head>
<body>

<h1>Dataset Inventory Dashboard</h1>
<p class="subtitle">Last updated: {now} &nbsp;|&nbsp; Auto-refreshes every 10s (re-run script to update data)</p>

<!-- Summary bar -->
<div class="summary-bar">
  <div class="summary-card">
    <div class="number">{total_existing_patients}</div>
    <div class="label">Labeled Patients</div>
  </div>
  <div class="summary-card">
    <div class="number">{total_existing_segments:,}</div>
    <div class="label">Existing Segments</div>
  </div>
  <div class="summary-card">
    <div class="number">{total_harvested:,}</div>
    <div class="label">Harvested Candidates</div>
  </div>
  <div class="summary-card">
    <div class="number">{total_all:,}</div>
    <div class="label">Total All Sources</div>
  </div>
</div>

<!-- Main histogram -->
<div class="section">
  <h2>Segments by Pattern Type</h2>
  <div class="bar-chart">
"""

    max_val = max(bar_values) if max(bar_values) > 0 else 1
    for i, (cat, val, color) in enumerate(zip(bar_categories, bar_values, bar_colors)):
        h = int((val / max_val) * 220) if val > 0 else 2
        val_class = ' class="zero"' if val == 0 else ""
        html += f"""    <div class="bar-col">
      <span class="bar-val"{val_class}>{val}</span>
      <div class="bar" style="height:{h}px; background:{color};"></div>
      <span class="bar-label">{cat}</span>
    </div>
"""

    html += """  </div>
</div>

<!-- Label status table -->
<div class="section">
  <h2>Label Status by Pattern Type</h2>
  <table>
    <tr>
      <th>Type</th>
      <th>N Patients</th>
      <th>N Segments</th>
      <th>Frequency Labels</th>
      <th>Discharge Timing</th>
      <th>Spatial Labels</th>
      <th>Laterality Labels</th>
    </tr>
"""

    type_colors = {
        "lpd": "#2166ac",
        "gpd": "#b2182b",
        "lrda": "#1b7837",
        "grda": "#762a83",
    }
    for st in ["lpd", "gpd", "lrda", "grda"]:
        color = type_colors[st]
        html += f"""    <tr>
      <td><span class="type-badge" style="background:{color}33; color:{color}; border:1px solid {color}66;">{st.upper()}</span></td>
      <td>{existing[st]}</td>
      <td>{existing_segs[st]}</td>
      <td>{freq_by_type[st]}</td>
      <td>{dt_by_type[st]}</td>
      <td>{ci_by_type[st]}</td>
      <td>{lat_by_type[st]}</td>
    </tr>
"""

    total_freq = sum(freq_by_type.values())
    total_dt = sum(dt_by_type.values())
    total_ci = sum(ci_by_type.values())
    total_lat = sum(lat_by_type.values())
    html += f"""    <tr style="font-weight:600; border-top:2px solid #444;">
      <td>TOTAL</td>
      <td>{total_existing_patients}</td>
      <td>{total_existing_segments}</td>
      <td>{total_freq}</td>
      <td>{total_dt}</td>
      <td>{total_ci}</td>
      <td>{total_lat}</td>
    </tr>
  </table>
</div>

<!-- Frequency distribution -->
<div class="section">
  <h2>Frequency Distribution (gold_standard_freq, N={len(freqs)})</h2>
  <div class="freq-chart">
"""

    max_fh = max(freq_hist) if max(freq_hist) > 0 else 1
    for bl, fh in zip(bin_labels, freq_hist):
        h = int((fh / max_fh) * 130) if fh > 0 else 2
        html += f"""    <div class="freq-col">
      <span class="freq-val">{fh}</span>
      <div class="freq-bar" style="height:{h}px;"></div>
      <span class="freq-label">{bl} Hz</span>
    </div>
"""

    html += f"""  </div>
</div>

<!-- Active harvests -->
<div class="section">
  <h2>Active Harvests</h2>
  <div class="harvest-grid">
    <div class="harvest-card">
      <h3>LPD Harvest (morgoth1)</h3>
      <div class="harvest-stat">{harvest_lpd}</div>
      <div class="harvest-detail">candidates from LPD folders</div>
    </div>
    <div class="harvest-card">
      <h3>Seizure LPD Harvest</h3>
      <div class="harvest-stat">{harvest_highfreq}</div>
      <div class="harvest-detail">high-frequency LPD candidates (>2.5 Hz)</div>
    </div>
    <div class="harvest-card">
      <h3>BIPD Harvest (morgoth2)</h3>
      <div class="harvest-stat {"zero" if bipd_count == 0 else ""}">{bipd_count}</div>
      <div class="harvest-detail">{"awaiting harvest" if bipd_count == 0 else "downloaded"}</div>
    </div>
    <div class="harvest-card">
      <h3>LRDA/GRDA Harvest (morgoth1)</h3>
      <div class="harvest-stat {"zero" if rda_lrda + rda_grda == 0 else ""}">{rda_lrda + rda_grda}</div>
      <div class="harvest-detail">{f"{rda_lrda} LRDA, {rda_grda} GRDA" if rda_lrda + rda_grda > 0 else "awaiting harvest"}</div>
    </div>
    <div class="harvest-card">
      <h3>Other / Controls (morgoth1)</h3>
      <div class="harvest-stat {"zero" if other_count == 0 else ""}">{other_count}</div>
      <div class="harvest-detail">{f"{other_count} control segments (no PD/RDA)" if other_count > 0 else "awaiting harvest"}</div>
    </div>
  </div>
</div>

<p style="text-align:center; color:#555; font-size:11px; margin-top:16px;">
  Re-run: <code style="color:#888;">conda run -n foe python code/generate_dataset_dashboard.py</code><br>
  Auto-update: <code style="color:#888;">while true; do conda run -n foe python code/generate_dataset_dashboard.py; sleep 30; done</code>
</p>

</body>
</html>
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print(f"Dashboard written to {OUT}")
    print(f"  Existing: {total_existing_patients} patients, {total_existing_segments} segments")
    print(f"  Harvested: {total_harvested} candidates")
    print(f"  Total: {total_all}")


if __name__ == "__main__":
    main()
