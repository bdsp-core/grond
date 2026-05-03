"""
Regenerate test case images from cached raw EEG segments in test_case_images/raw_eeg/.
Does not require the external drive or Excel file.
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from mne.filter import notch_filter, filter_data
import scipy.io
import warnings

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))
sys.path.insert(0, str(CODE_DIR / 'rda_detector'))

from pd_detect_alternate import pd_detect_alternate
from rda1b_fft import rda1b_fft
from browse_results import BIPOLAR_CHANNELS, get_bipolar
from generate_test_images import run_detector, draw_figure, draw_pointiness_figure

OUTPUT_DIR = CODE_DIR.parent / 'test_case_images'
RAW_DIR = OUTPUT_DIR / 'raw_eeg'


def main():
    mat_files = sorted(RAW_DIR.glob('*.mat'))
    print(f'Found {len(mat_files)} cached segments')

    metadata = {}

    for i, f in enumerate(mat_files):
        name = f.stem
        ptype = name.split('_', 2)[0]
        print(f'  [{i+1}/{len(mat_files)}] {name}', end='', flush=True)

        try:
            mat = scipy.io.loadmat(str(f))
            data = mat['data']
            fs = int(mat['Fs'].ravel()[0])
            if data.shape[0] > data.shape[1]:
                data = data.T

            seg_filtered = notch_filter(data.astype(float), fs, 60, n_jobs=1, verbose="ERROR")
            seg_filtered = filter_data(seg_filtered, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
            seg_bi = get_bipolar(seg_filtered)

            result_row = run_detector(data, fs, ptype)
            result_row['files'] = name

            # Collect metadata for viewer
            freq_a = result_row.get('event_frequency', np.nan)
            freq_b = result_row.get('acf_frequency', np.nan)
            metadata[name] = {
                'freq_a': round(freq_a, 2) if np.isfinite(freq_a) else None,
                'freq_b': round(freq_b, 2) if np.isfinite(freq_b) else None,
                'type': str(result_row.get('type_event', '')),
                'expert_freq': None,
            }

            # Main image
            fig = draw_figure(result_row, seg_bi, fs, ptype, title_extra=name)
            fig.savefig(str(OUTPUT_DIR / f'{name}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Pointiness image
            fig2 = draw_pointiness_figure(result_row, seg_bi, fs, ptype, title_extra=name)
            fig2.savefig(str(OUTPUT_DIR / f'{name}_pointiness.png'), dpi=150, bbox_inches='tight')
            plt.close(fig2)

            print('  ✓')
        except Exception as e:
            print(f'  FAILED: {e}')

    # Write metadata.json and metadata.js
    meta_json_path = OUTPUT_DIR / 'metadata.json'
    meta_js_path = OUTPUT_DIR / 'metadata.js'
    with open(str(meta_json_path), 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(str(meta_js_path), 'w') as f:
        f.write('metadata = ')
        json.dump(metadata, f)
        f.write('; show();')
    print(f'Metadata written to {meta_json_path}')

    print(f'\nDone! {len(mat_files)} image pairs saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
