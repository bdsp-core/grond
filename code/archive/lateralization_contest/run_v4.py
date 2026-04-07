#!/usr/bin/env python3
"""Run V4 lateralization contest methods.

Usage:
    python code/lateralization_contest/run_v4.py batch1   # methods 1-5
    python code/lateralization_contest/run_v4.py batch2   # methods 6-10
    ...
"""
import sys
import importlib
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from lateralization_contest.harness_v4 import (
    load_contest_data, run_method, evaluate, save_result, update_html_leaderboard
)
from lateralization_contest.base import LateralMethod


def get_all_methods():
    """Import all LateralMethod subclasses from the method files."""
    methods = []
    for mod_name in ['methods_power', 'methods_rhythm', 'methods_fit',
                     'methods_waveform', 'methods_advanced',
                     'methods_v4_unified', 'methods_v4_unified2']:
        try:
            mod = importlib.import_module(f'lateralization_contest.{mod_name}')
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (isinstance(attr, type) and issubclass(attr, LateralMethod)
                        and attr is not LateralMethod):
                    methods.append(attr())
        except Exception as e:
            print(f"Warning: could not load {mod_name}: {e}")
    return methods


BATCHES = {
    'batch1': ['L01_DeltaBandpower', 'L02_NarrowbandPeak', 'L03_PeakToMeanRatio',
                'L04_BandpowerRatio', 'L05_RMSAmplitude'],
    'batch2': ['L06_ACFPeak', 'L07_HilbertCV', 'L08_SpectralConcentration',
                'L09_ZeroCrossingRegularity', 'L10_EnvelopeRegularity'],
    'batch3': ['L11_VarExplained', 'L12_TemplateMatch', 'L13_AR2Periodicity',
                'L14_NarrowbandVE', 'L15_MultiChannelVE'],
    'batch4': ['L16_AmplitudeConsistency', 'L17_PeakRegularity', 'L18_WaveformSymmetry',
                'L19_PeakToTrough', 'L20_Kurtosis'],
    'batch5': ['L21_IntraHemiCoherence', 'L22_IntraHemiPLV', 'L23_InterHemiCorr',
                'L24_EnvelopeAmplitude', 'L25_SVDDominance'],
    'batch6': ['U01_HilbertCV', 'U02_ACFPeakFreq', 'U03_SpectralPeak',
                'U04_VarExplained', 'U05_IPIRegularity'],
    'batch7': ['U06_EnvAmp_HilbertFreq', 'U07_RMS_ACFFreq', 'U08_BP_SpectralFreq',
                'U09_NarrowbandVE', 'U10_MultiCh_HilbertFreq'],
    'batch8': ['U11_HilbertCV_Top3', 'U12_EnvAmp_VEFreq', 'U13_PLV_HilbertFreq',
                'U14_BP_IPIFreq', 'U15_EnvAmp_ACFFreq'],
    'batch9': ['V01_DomHemi_Top3Hilbert', 'V02_PowerWeightedHilbert', 'V03_ConsistencySelected',
                'V04_PLVSelected', 'V05_AdaptiveTopK'],
    'batch10': ['V06_MultiMethodFreq', 'V07_NarrowbandSweep', 'V08_TemplateBank',
                 'V09_VEGrid_EnvScore', 'V10_CepstralFreq'],
    'batch11': ['V11_NarrowbandAtPeak', 'V12_IterativeRefine', 'V13_MatchedFilterLat',
                 'V14_FreqSpecificPowerRatio', 'V15_CrossFreqProfile'],
    'batch12': ['V16_EnvPeriodicity', 'V17_PeakCountFreq', 'V18_ZeroCrossFreq',
                 'V19_SpatialCoherenceFreq', 'V20_SVDFreq'],
    'batch13': ['V21_ChannelFreqMatrix', 'V22_EnvAmp_DomHilbert', 'V23_CherryPick',
                 'V24_SoftChannelWeight', 'V25_FreqBandEnvRatio'],
    'batch14': ['W01_DomOnly_StrictHilbert', 'W02_DomOnly_AutoK', 'W03_DomOnly_QualityWeighted',
                 'W04_DomOnly_MultiMethod', 'W05_DomOnly_IterRefine'],
    'batch15': ['W06_AutoChannel_EnvThreshold', 'W07_AutoChannel_FreqAgreement',
                 'W08_DomOnly_VEFreq', 'W09_DomOnly_IPIFreq', 'W10_DomOnly_EnvPeakFreq'],
    'all': None,
}


def main():
    batch_name = sys.argv[1] if len(sys.argv) > 1 else 'all'
    filter_names = BATCHES.get(batch_name)

    print(f"V4 Contest — batch '{batch_name}'")
    print("Loading data...")
    data = load_contest_data(verbose=True)

    all_methods = get_all_methods()
    if filter_names:
        methods = [m for m in all_methods if m.name in filter_names]
    else:
        methods = all_methods

    print(f"\nRunning {len(methods)} methods:")
    for m in methods:
        print(f"  - {m.name}")

    for method in methods:
        result_file = Path(f'results/lateralization_contest_v4/{method.name}.json')
        if result_file.exists():
            print(f"\n{method.name}: CACHED")
            continue

        print(f"\n{'=' * 50}")
        print(f"Running: {method.name}")
        results = run_method(method, data, verbose=True)
        metrics = evaluate(results, data)
        save_result(method.name, metrics)
        auc = metrics.get('primary_auc', 0)
        marker = ' ***' if auc and auc > 0.65 else ' **' if auc and auc > 0.60 else ''
        print(f"  AUC: {metrics['primary_auc']}  Cohen's d: {metrics['cohens_d']}{marker}")
        update_html_leaderboard()

    print("\nDone!")
    update_html_leaderboard()


if __name__ == '__main__':
    main()
