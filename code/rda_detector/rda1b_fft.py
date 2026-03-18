from fooof import FOOOF
from joblib import Parallel, delayed
from mne.filter import notch_filter,filter_data
import hdf5storage as hs
import scipy.io as sio
import numpy as np
import pandas as pd
import numpy.matlib
import math
import os
import pdb
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from collections import Counter
import zlib

# global var
bipolar_channels=['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2','Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4',
                 'F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz']
mono_channels=['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2','EKG']
fooofgap=np.empty((1,5))
fooofgap[:]=np.nan
fooofgap=fooofgap[0]
thr_bw=0.5
freq_range=[.5,4]

# callbacks 
def fcn_computeSpectra(x,Fs):
    """
    Calculate signal spectrum using Fourier Transform
    Attributes:
        x (numpy array) : input signals
        Fs (numpy) : sampling frequency
    Output:
        psdx (numpy array): power spectrum (non-logged)
    """ 
    x = x-np.mean(x)
    N=len(x)
    #xdft=np.fft.fft(x)
    #xdft=xdft[0:int(N/2)]
    #psdx=(1/(Fs*N))*np.square(abs(xdft))
    #psdx[1:-2]=2*psdx[1:-2]

    xdft = np.fft.fft(x)
    xdft = xdft[:len(xdft) //2]
    psdx = (1/(Fs*N))*np.square(np.abs(xdft))

    return psdx

def fcn_getBanana(X):
    """
    Apply a differential longitudinal montage to EEG data
    Attributes:
        X (numpy array): 19 raw EEG channels
    Output:
        bipolar_data (numpy array): re-referenced EEG data
    """ 
    bipolar_ids = np.array([[mono_channels.index(bc.split('-')[0]),mono_channels.index(bc.split('-')[1])] for bc in bipolar_channels])
    bipolar_data = X[bipolar_ids[:,0]]-X[bipolar_ids[:,1]]
    return bipolar_data

def fcn_rdafooof_enhanced(seg,freqs, Fs,channel_filter):
    """
    Apply fooof model on the channel spectra
    Attributes:
        seg (numpy array): 18 EEG channels
        freqs (numpy array): 19 spectra of EEG channels
        Fs (numpy) : sampling frequency
        channel filter (boolean) : 0/1 - apply a channel selector based on valid EEG data or noise
    Output:
        fooof_ (numpy array): peaks, location and bw of frequency peaks that meet rda condition
        channels (numpy array): list of channels where an rda is detected
        spectra (numpy array): list of spectra for channels where an rda is detected
        rda_scores (numpy array): per-channel RDA score (peak_power / baseline_power), NaN if no peak
        channel_freqs (numpy array): per-channel detected peak frequency, NaN if no peak
    """
    fooof=[]
    count = 0
    # Track which original channel index maps to each fooof row
    channel_indices = []
    spectra = np.zeros([len(seg),int(seg.shape[1]/2)])
    for k in range(0,len(seg)):

        if channel_filter:
            x = seg[k]
            sig_range = np.max(abs(x))-np.min(abs(x))
            if np.var(x)>50*sig_range:
                continue


        spectrum=fcn_computeSpectra(seg[k],Fs)
        spectra[count,:] = spectrum
        #spectrum=s[0]
        #spectrum=s
        fm = FOOOF()
        freqs = np.array(freqs)
        spectrum = np.array(spectrum)

        try:
            fm.fit(freqs, spectrum, freq_range)
        except Exception as e:
            print(f"Error fitting FOOOF model for segment {k+1}: {e}")
            return 69420, 69420, 69420, np.full(len(seg), np.nan), np.full(len(seg), np.nan)


        tmp=fm.peak_params_

        if len(tmp)==0:
            x=fooofgap
            fooof.append(x)
        else:
            x=np.append(np.append(tmp[0],fm.error_),fm.r_squared_)
            fooof.append(x)
        channel_indices.append(k)
        count+=1
    fooof=np.array(fooof)
    bw=np.round(100*fooof[:,2])/100

    # Initialize per-channel scores (all 18 channels)
    rda_scores = np.full(len(seg), np.nan)
    channel_freqs = np.full(len(seg), np.nan)

    idx=np.where(bw<=thr_bw)
    idx2 = []
    pks = fooof[:,0]
    bw = fooof[:,2]

    for pk_idx,pk in enumerate(pks):
        closest_value = min(freqs, key=lambda x: abs(x-pk))

        #select the indexes corresponding to the detected peak as well as the bandwidth identified around it - this all would represent the freq content of interest
        select_pk_bw = (freqs > closest_value-(bw[pk_idx]/2)) & (freqs < closest_value+(bw[pk_idx]/2))

        #select the rest of the signal in the delta band [0.5, 3] Hz and everything outside the peak/content of interest - calculate the mean
        baseline_power = np.mean(spectra[pk_idx,(freqs<=3) & (freqs>=0.5) & ~select_pk_bw])
        peak_power = np.mean(spectra[pk_idx,select_pk_bw])

        # Store continuous score and frequency for this channel
        orig_ch_idx = channel_indices[pk_idx]
        if baseline_power > 0 and not np.isnan(pk):
            rda_scores[orig_ch_idx] = peak_power / baseline_power
            channel_freqs[orig_ch_idx] = pk

        if peak_power > baseline_power:
            idx2.append(pk_idx)

    if len(idx2)==0:
        return 69420, 69420, 69420, rda_scores, channel_freqs

    idx2 = (np.array(idx2),)

    fooof_=fooof[idx2]

    channels_=np.array(bipolar_channels)[idx2[0]]

    return fooof_,channels_,spectra, rda_scores, channel_freqs


def rda1b_fft(segment,fs,channel_filter):
    """
    Calculate frequency of event, spatial extent and spatial areas for RDA using the algorithm rda1_fft
    Attributes:
        segment (numpy array): 18 EEG channel segments
        fs (numpy) : sampling frequency
        channel filter (boolean) : 0/1 - apply a channel selector based on valid EEG data or noise
    Output:
        data_obj (panda series): 
            type_event: LRDA or GRDA based on the spatial extent value
            event_frequency: median frequency of events across all detected channels
            power: median event power
            bandwidth: median peak bandwidth of identified peaks
            fit_error: median fooof fit error across channels
            r_squared: median r-squared for fooof model
            spatial_extent: 0 - no channels show rda, 1 - all channels show rda
            spatial_areas: spatial areas identified based on the channels selected - LF, RF, LCP, RCP, RT, LT, LO, RO
            channels: number of channels detected to have RDA events

    """ 
    # filters to denoise
    segment=notch_filter(segment,fs,60,n_jobs=1,verbose="ERROR")
    segment=filter_data(segment,fs,0.5,40,n_jobs=1,verbose="ERROR")

    # L-bipolar
    segment=fcn_getBanana(segment)
    segment=np.array(segment)

    seg=segment#[:16,:]
    
    #N=len(seg[0])     
    #freqs1=np.arange(0,fs/2,fs/N) 
    freqs = np.fft.fftfreq(seg.shape[1],1/fs)
    freqs = freqs[:len(freqs) //2]
    #pdb.set_trace()
    # Left/right hemisphere channel indices (into bipolar_channels, 0-indexed)
    left_indices = [0, 1, 2, 3, 8, 9, 10, 11]   # Fp1-F7, F7-T3, T3-T5, T5-O1, Fp1-F3, F3-C3, C3-P3, P3-O1
    right_indices = [4, 5, 6, 7, 12, 13, 14, 15] # Fp2-F8, F8-T4, T4-T6, T6-O2, Fp2-F4, F4-C4, C4-P4, P4-O2
    # Midline indices 16, 17 (Fz-Cz, Cz-Pz) excluded from laterality

    fooof,channels,spectra, rda_scores, channel_freqs=fcn_rdafooof_enhanced(seg,freqs, fs,channel_filter)

    # Build per-channel score dicts (always available, even if no RDA detected)
    channel_rda_scores = {bipolar_channels[i]: rda_scores[i] for i in range(len(bipolar_channels))}
    channel_frequencies = {bipolar_channels[i]: channel_freqs[i] for i in range(len(bipolar_channels))}

    # Region-level scores (mean RDA score per region)
    # Use score=1.0 (neutral) for channels with no detected peak
    scores_for_lat = np.where(np.isnan(rda_scores), 1.0, rda_scores)
    region_channel_map = {
        'LF': [0, 8, 9, 1],    # Fp1-F7, Fp1-F3, F3-C3, F7-T3
        'RF': [4, 5, 12, 13],  # Fp2-F8, F8-T4, Fp2-F4, F4-C4
        'LT': [2, 3],          # T3-T5, T5-O1
        'RT': [6, 7],          # T4-T6, T6-O2
        'LCP': [10],           # C3-P3
        'RCP': [14],           # C4-P4
        'LO': [11],            # P3-O1
        'RO': [15],            # P4-O2
    }
    region_scores = {}
    for region, idxs in region_channel_map.items():
        region_vals = scores_for_lat[idxs]
        region_scores[region] = float(np.mean(region_vals))

    # Compute laterality index from region means (equal weight per region,
    # avoids over-weighting frontal regions which have 4 channels vs 1 for occipital)
    left_mean = np.mean([region_scores['LF'], region_scores['LT'],
                         region_scores['LCP'], region_scores['LO']])
    right_mean = np.mean([region_scores['RF'], region_scores['RT'],
                          region_scores['RCP'], region_scores['RO']])
    denom = right_mean + left_mean
    if denom > 0:
        laterality_index = (right_mean - left_mean) / denom
    else:
        laterality_index = 0.0

    spatial_areas = []
    if np.any(fooof == 69420) or np.any(channels == 69420):
        data_obj = {
            "type_event":np.nan,
            "event_frequency": np.nan,
            "power": np.nan,
            "bandwidth": np.nan,
            "fit_error": np.nan,
            "r_squared": np.nan,
            "spatial_extent": np.nan,
            "spatial_areas":np.nan,
            "channels": np.nan,
            "channel_rda_scores": channel_rda_scores,
            "channel_frequencies": channel_frequencies,
            "region_scores": region_scores,
            "laterality_index": laterality_index,
            "left_mean_score": left_mean,
            "right_mean_score": right_mean,
        }
    else:

        if len(channels)==0:
            data_obj = {
                "type_event":np.nan,
                "event_frequency": np.nan,
                "power": np.nan,
                "bandwidth": np.nan,
                "fit_error": np.nan,
                "r_squared": np.nan,
                "spatial_extent": np.nan,
                "spatial_areas":np.nan,
                "channels": np.nan,
                "channel_rda_scores": channel_rda_scores,
                "channel_frequencies": channel_frequencies,
                "region_scores": region_scores,
                "laterality_index": laterality_index,
                "left_mean_score": left_mean,
                "right_mean_score": right_mean,
            }
        else:
            for channel in channels:
                if channel in ['Fp1-F7','Fp1-F3','F3-C3','F7-T3']:
                    spatial_areas.append('LF')
                if channel in ['Fp2-F8','F8-T4','Fp2-F4','F4-C4']:
                    spatial_areas.append('RF')
                if channel in ['C3-P3']:
                    spatial_areas.append('LCP')
                if channel in ['C4-P4']:
                    spatial_areas.append('RCP')
                if channel in ['T3-T5','T5-O1']:
                    spatial_areas.append('LT')
                if channel in ['T4-T6','T6-O2']:
                    spatial_areas.append('RT')
                if channel in ['P3-O1']:
                    spatial_areas.append('LO')
                if channel in ['P4-O2']:
                    spatial_areas.append('RO')

            spatial_areas = list(set(spatial_areas))

            if len(channels)/seg.shape[0] > 0.8:
                type_event = "GRDA"
            else:
                type_event = "LRDA"

            data_obj = {
                "type_event":type_event,
                "event_frequency": np.median(fooof[:,0]),
                "power": np.median(fooof[:,1]),
                "bandwidth": np.median(fooof[:,2]),
                "fit_error": np.median(fooof[:,3]),
                "r_squared": np.median(fooof[:,4]),
                "spatial_extent": len(channels)/seg.shape[0],
                "spatial_areas":spatial_areas,
                "channels": pd.Series(channels),
                "channel_rda_scores": channel_rda_scores,
                "channel_frequencies": channel_frequencies,
                "region_scores": region_scores,
                "laterality_index": laterality_index,
                "left_mean_score": left_mean,
                "right_mean_score": right_mean,
                }

    return data_obj,spectra, freqs

