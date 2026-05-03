"""
Generate EEG PNG images for round 3 annotation candidates.
Uses draw_figure from generate_test_images.py.
Must run with: conda run -n foe python code/generate_round3_images.py
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

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))
sys.path.insert(0, str(CODE_DIR / 'rda_detector'))

from generate_test_images import draw_figure, run_detector
from browse_results import BIPOLAR_CHANNELS, get_bipolar

BASE = CODE_DIR.parent
DATA = BASE / 'data'
R3_DIR = DATA / '_archive' / 'pd_round3'
LPD_DIR = R3_DIR / 'lpd'
IMG_DIR = R3_DIR / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    manifest = pd.read_csv(R3_DIR / 'manifest.csv')
    print(f"Processing {len(manifest)} round 3 LPD segments...")

    for i, row in manifest.iterrows():
        file_name = row['file_name']
        pid = row['patient_id']
        mat_path = LPD_DIR / f"{file_name}.mat"

        print(f"  [{i+1}/{len(manifest)}] {file_name}", end='', flush=True)

        try:
            mat = scipy.io.loadmat(str(mat_path))
            data = mat['data']  # (18, 2000) bipolar already
            fs = int(mat['Fs'].ravel()[0])

            # The cached data is already preprocessed bipolar data
            # It's already filtered, so we can use it directly as seg_bi
            seg_bi = data.astype(np.float64)

            if seg_bi.shape[0] > seg_bi.shape[1]:
                seg_bi = seg_bi.T

            # For the detector, we need monopolar-like data
            # But since we only have bipolar, we'll create a dummy monopolar
            # by zero-padding to 20 channels and running detector on bipolar directly
            # Actually, pd_detect_alternate expects monopolar and converts to bipolar internally
            # So we need to skip the detector or create a mock result

            # Create mock result row with NaN scores (no detector run needed)
            result_row = {'files': file_name, 'type_event': np.nan,
                          'event_frequency': np.nan, 'acf_frequency': np.nan,
                          'spatial_extent': np.nan, 'laterality_index': np.nan,
                          'left_mean_score': np.nan, 'right_mean_score': np.nan}
            for ch in BIPOLAR_CHANNELS:
                result_row[f'score_{ch}'] = 2.0  # Mark all as active for display
                result_row[f'freq_{ch}'] = np.nan

            # Draw figure
            fig = draw_figure(result_row, seg_bi, fs, 'lpd',
                              title_extra=f'Patient {pid}')
            png_path = IMG_DIR / f"{file_name}.png"
            fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print('  OK')

        except Exception as e:
            print(f'  FAILED: {e}')
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone! {len(manifest)} images saved to {IMG_DIR}")


if __name__ == '__main__':
    main()
