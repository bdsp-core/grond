#!/usr/bin/env python3
"""LRDA laterality disagreement triage.

Identifies the LRDA segments where the W05 (V1/V12) laterality call
disagrees with the consensus laterality (>=2 of 3 raters agree on
left/right; per-rater accept required). For each disagreement, dumps:

    - mat_file, patient_id, est_freq
    - consensus laterality + each rater's individual L/R call
    - algorithm's call + the 16 laterality features
    - which features point left vs right (so we can see whether the
      mistake is robust across discriminators or hinges on one signal)

Also renders a multi-panel EEG figure (bipolar 18-ch, 10 s) annotated
with the rater calls + feature dump per case.

Output:
    paper_materials/independent_expert_tasks/lrda/laterality_disagreements.md
    paper_materials/independent_expert_tasks/lrda/laterality_disagreements_eeg.png

    conda run -n morgoth python code/evaluation/lrda_laterality_disagreement_triage.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))
from generate_rda_freq_labeler import load_segment, FS, LEFT_CHS, RIGHT_CHS  # type: ignore

BIPOLAR_LABELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'
RAW_DIR = LABELS_DIR / 'raw_inputs' / 'independent_expert_v1'
FEAT_CSV = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_features.csv'
OUT_MD = TASKS_DIR / 'laterality_disagreements.md'
OUT_PNG = TASKS_DIR / 'laterality_disagreements_eeg.png'


def load_status_and_lat():
    status = {r: {} for r in ('MW', 'SZ', 'TZ')}
    files = [
        ('TZ/lrda_freq_labeling_results_TZ.json', 'TZ'),
        ('SZ/rda_freq_labeling_results-2.json', 'SZ'),
        ('MW/rda_freq_labeling_results-mbw-update20.json', 'MW'),
    ]
    for rel, rater in files:
        with open(RAW_DIR / rel) as f:
            d = json.load(f)
        for v in d.values():
            mf = v.get('mat_file')
            sub = (v.get('subtype') or '').lower()
            if mf and sub == 'lrda':
                status[rater][mf] = v.get('action') or 'unknown'

    lat = {r: {} for r in ('MW', 'SZ', 'TZ')}
    with open(LABELS_DIR / 'labels.csv') as f:
        for row in csv.DictReader(f):
            r = row['rater']
            if r not in ('MW', 'SZ', 'TZ'):
                continue
            if row['label_type'] != 'laterality':
                continue
            v = row['value'].strip().lower()
            if v in ('left', 'right'):
                lat[r][row['mat_file']] = v
    return status, lat


def algo_v12_call_from_features(feat_row):
    """V12 (and V1) laterality is: pass2_env_log_ratio > 0  =>  left, else right."""
    return 'left' if float(feat_row['pass2_env_log_ratio']) > 0 else 'right'


def consensus_lat(mf, status, lat):
    votes = []
    for r in ('MW', 'SZ', 'TZ'):
        if status[r].get(mf) == 'accept' and mf in lat[r]:
            votes.append(lat[r][mf])
    if len(votes) < 2:
        return None
    c = Counter(votes)
    top, count = c.most_common(1)[0]
    return top if count >= 2 else None


def main():
    status, lat = load_status_and_lat()
    with open(FEAT_CSV) as f:
        feat_rows = {r['mat_file']: r for r in csv.DictReader(f)}

    rows = []
    for mf, fr in feat_rows.items():
        cons = consensus_lat(mf, status, lat)
        if cons is None:
            continue
        algo = algo_v12_call_from_features(fr)
        if algo != cons:
            rows.append({
                'mat_file': mf,
                'patient_id': fr.get('patient_id', ''),
                'consensus': cons,
                'algo': algo,
                'mw_status': status['MW'].get(mf, 'absent'),
                'sz_status': status['SZ'].get(mf, 'absent'),
                'tz_status': status['TZ'].get(mf, 'absent'),
                'mw_lat': lat['MW'].get(mf, '-'),
                'sz_lat': lat['SZ'].get(mf, '-'),
                'tz_lat': lat['TZ'].get(mf, '-'),
                'feats': fr,
            })
    print(f'Found {len(rows)} consensus-laterality disagreement(s) (algo vs >=2-of-3 consensus).')

    # Order by est_freq for readability
    rows.sort(key=lambda r: float(r['feats']['est_freq']))

    # ------- Markdown report -------
    md_lines = []
    md_lines.append('# LRDA laterality disagreements: algorithm vs consensus')
    md_lines.append('')
    md_lines.append(f'Found **{len(rows)}** segments where the V12/V1 algorithm laterality call '
                    f'disagrees with the >=2-of-3 majority-accept consensus laterality '
                    f'(consensus dataset = 155 segments).')
    md_lines.append('')
    md_lines.append('Algorithm rule: `pass2_env_log_ratio > 0` means left dominant. '
                    'A segment ends up wrong when several discriminators disagree, when the '
                    'estimated frequency is wrong (so the pass-2 narrowband filter is '
                    'centered on the wrong rhythm), or when laterality is genuinely '
                    'ambiguous.')
    md_lines.append('')
    md_lines.append('Columns:')
    md_lines.append('- `pass1_var`: log(L/R) of pass-1 broadband variance (sign agrees with consensus if positive=left). '
                    '`pass2_env`: dominant W05 discriminator. `nb_var`: pass-2 narrowband variance ratio. '
                    '`top3_var`: top-3-channel variance ratio. `peak_prom`: spectral peak prominence ratio. '
                    '`max_ch`: log of max-single-channel variance ratio.')
    md_lines.append('- `agree_p1p2`: 1 if pass-1 and pass-2 picked the same side. '
                    '`agree_top3`: 1 if top-3 and uniform-mean agree. `if_cv`: Hilbert IF coefficient of variation '
                    '(estimator confidence; high = unreliable). `art_l`/`art_r`: low-freq drift artifact score per side.')
    md_lines.append('')

    # Header
    md_lines.append('| # | mat_file (short) | freq | consensus | algo | MW | SZ | TZ | pass1_var | pass2_env | nb_var | top3_var | peak_prom | max_ch | agree_p1p2 | agree_top3 | if_cv | art_l | art_r |')
    md_lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for i, r in enumerate(rows):
        f = r['feats']
        short = r['mat_file'].replace('.mat', '').replace('sub-', '')[:30]
        # Sign-aware: show with explicit + so we can tell which side each feature points to
        def sgn(x):
            try:
                v = float(x)
                return f'{v:+.2f}'
            except Exception:
                return str(x)
        md_lines.append(
            f"| {i+1} | `{short}` | {float(f['est_freq']):.2f} | **{r['consensus']}** | {r['algo']} | "
            f"{r['mw_lat']} | {r['sz_lat']} | {r['tz_lat']} | "
            f"{sgn(f['pass1_var_log_ratio'])} | "
            f"{sgn(f['pass2_env_log_ratio'])} | "
            f"{sgn(f['narrowband_var_log_ratio'])} | "
            f"{sgn(f['top3_var_log_ratio'])} | "
            f"{sgn(f['spectral_peak_prom_log_ratio'])} | "
            f"{sgn(f['lr_max_ch_log_ratio'])} | "
            f"{int(float(f['pass1_pass2_agreement']))} | "
            f"{int(float(f['top3_uniform_agreement']))} | "
            f"{float(f['est_freq_if_cv']):.2f} | "
            f"{float(f['left_artifact_score']):.2f} | "
            f"{float(f['right_artifact_score']):.2f} |"
        )

    md_lines.append('')
    md_lines.append('Sign convention for the log-ratio features: positive = left dominant.')
    md_lines.append('')

    md_lines.append('## Per-case interpretation prompts')
    md_lines.append('For each row above, judge:')
    md_lines.append('')
    md_lines.append('1. **Are the discriminators unanimous in being wrong, or split?** If split, the rule could be '
                    'made more robust by combining (e.g., majority vote across pass1/pass2/peak_prom/max_ch) instead '
                    'of relying solely on `pass2_env_log_ratio`.')
    md_lines.append('2. **Is the est_freq correct?** If the pass-2 narrowband filter is centered on a wrong frequency, '
                    'the envelope ratio measures the wrong rhythm. Check `est_freq` against the visual rhythm in the '
                    'EEG figure.')
    md_lines.append('3. **Is laterality genuinely binary?** If the pattern is bilateral with subtle asymmetry, the '
                    'binary L/R call is ill-defined and the algorithm gets penalized by the binary metric.')
    md_lines.append('4. **Is one rater the outlier?** If two raters and the algo agree, the consensus is fragile; '
                    'a 4-way consensus would tighten the ground truth.')
    md_lines.append('')

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text('\n'.join(md_lines))
    print(f'Wrote {OUT_MD.relative_to(PROJECT_DIR)}')

    # ------- EEG figure -------
    if not rows:
        print('No disagreements; skipping EEG figure.')
        return

    n = len(rows)
    fig_h = max(4, 3.0 * n)
    fig, axes = plt.subplots(n, 1, figsize=(13, fig_h), squeeze=False)
    for i, r in enumerate(rows):
        ax = axes[i, 0]
        seg = load_segment(r['mat_file'])
        if seg is None:
            ax.text(0.5, 0.5, f'Could not load {r["mat_file"]}', ha='center', va='center')
            continue
        n_ch, n_samp = seg.shape
        t = np.arange(n_samp) / FS
        # Stack 18 bipolar channels with vertical offset; left chans first then right.
        order = list(LEFT_CHS) + list(RIGHT_CHS) + [c for c in range(n_ch) if c not in LEFT_CHS and c not in RIGHT_CHS]
        spacing = 200.0  # microvolts between channels
        for k, ch in enumerate(order):
            x = seg[ch] - np.median(seg[ch])
            x = np.clip(x, -spacing/2, spacing/2)
            y = -k * spacing
            ax.plot(t, x + y, 'k', linewidth=0.5)
            label = BIPOLAR_LABELS[ch] if ch < len(BIPOLAR_LABELS) else f'ch{ch}'
            color = '#1f77b4' if ch in LEFT_CHS else ('#d62728' if ch in RIGHT_CHS else 'k')
            ax.text(-0.15, y, label, ha='right', va='center', fontsize=7, color=color)
        ax.set_xlim(-0.4, t[-1] + 0.05)
        ax.set_ylim(-len(order) * spacing - spacing/2, spacing/2)
        ax.set_yticks([])
        if i == n - 1:
            ax.set_xlabel('time (s)')
        # Title with diagnostic block
        f = r['feats']
        title = (f"#{i+1}  {r['mat_file']}   "
                 f"est_freq={float(f['est_freq']):.2f} Hz  if_cv={float(f['est_freq_if_cv']):.2f}\n"
                 f"consensus={r['consensus']}   algo={r['algo']}   "
                 f"MW/SZ/TZ = {r['mw_lat']}/{r['sz_lat']}/{r['tz_lat']}   "
                 f"pass1={float(f['pass1_var_log_ratio']):+.2f}  "
                 f"pass2={float(f['pass2_env_log_ratio']):+.2f}  "
                 f"nb_var={float(f['narrowband_var_log_ratio']):+.2f}  "
                 f"top3={float(f['top3_var_log_ratio']):+.2f}  "
                 f"peak_prom={float(f['spectral_peak_prom_log_ratio']):+.2f}")
        ax.set_title(title, fontsize=9, loc='left')
        # Color band marking which side each call selects
        ax.axvspan(0, 0.05, ymin=0.5, ymax=1.0, color='#1f77b4', alpha=0.15, label='left chans')
        ax.axvspan(0, 0.05, ymin=0.0, ymax=0.5, color='#d62728', alpha=0.15, label='right chans')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'Wrote {OUT_PNG.relative_to(PROJECT_DIR)}')


if __name__ == '__main__':
    main()
