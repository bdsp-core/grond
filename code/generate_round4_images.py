"""
Generate EEG PNG images for round 4 annotation candidates (GPD + LPD).
Uses draw_figure from generate_test_images.py.
Must run with: conda run -n foe python code/generate_round4_images.py
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
from pathlib import Path
from mne.filter import notch_filter, filter_data
import warnings
warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))
sys.path.insert(0, str(CODE_DIR / 'rda_detector'))

from generate_test_images import draw_figure, run_detector
from browse_results import BIPOLAR_CHANNELS, get_bipolar

BASE = CODE_DIR.parent
DATA = BASE / 'data'
R4_DIR = DATA / '_archive' / 'annotation_round4'
GPD_DIR = R4_DIR / 'gpd'
LPD_DIR = R4_DIR / 'lpd'
IMG_DIR = R4_DIR / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)


def process_segment(row, subtype_dir, subtype_label):
    """Process a single segment: load .mat, generate image."""
    file_name = row['file_name']
    pid = row['patient_id']
    mat_path = subtype_dir / f"{file_name}.mat"

    try:
        mat = scipy.io.loadmat(str(mat_path))
        data = mat['data']  # (18, 2000) bipolar already
        fs = int(mat['Fs'].ravel()[0])

        # The cached data is already preprocessed bipolar data
        seg_bi = data.astype(np.float64)

        if seg_bi.shape[0] > seg_bi.shape[1]:
            seg_bi = seg_bi.T

        # Create mock result row with NaN scores (no detector run needed)
        result_row = {'files': file_name, 'type_event': np.nan,
                      'event_frequency': np.nan, 'acf_frequency': np.nan,
                      'spatial_extent': np.nan, 'laterality_index': np.nan,
                      'left_mean_score': np.nan, 'right_mean_score': np.nan}
        for ch in BIPOLAR_CHANNELS:
            result_row[f'score_{ch}'] = 2.0  # Mark all as active for display
            result_row[f'freq_{ch}'] = np.nan

        # Draw figure
        fig = draw_figure(result_row, seg_bi, fs, subtype_label,
                          title_extra=f'Patient {pid}')
        png_path = IMG_DIR / f"{file_name}.png"
        fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False


def main():
    manifest = pd.read_csv(R4_DIR / 'manifest.csv')
    print(f"Processing {len(manifest)} round 4 segments...")

    gpd_rows = manifest[manifest['subtype'] == 'gpd']
    lpd_rows = manifest[manifest['subtype'] == 'lpd']
    print(f"  GPD: {len(gpd_rows)}, LPD: {len(lpd_rows)}")

    success = 0
    fail = 0

    # Process GPD
    for i, (_, row) in enumerate(gpd_rows.iterrows()):
        print(f"  [{i+1}/{len(gpd_rows)}] GPD: {row['file_name']}", end='', flush=True)
        if process_segment(row, GPD_DIR, 'gpd'):
            print('  OK')
            success += 1
        else:
            print('  FAILED')
            fail += 1

    # Process LPD
    for i, (_, row) in enumerate(lpd_rows.iterrows()):
        print(f"  [{i+1}/{len(lpd_rows)}] LPD: {row['file_name']}", end='', flush=True)
        if process_segment(row, LPD_DIR, 'lpd'):
            print('  OK')
            success += 1
        else:
            print('  FAILED')
            fail += 1

    print(f"\nDone! {success} images saved, {fail} failed. Output: {IMG_DIR}")


if __name__ == '__main__':
    main()
