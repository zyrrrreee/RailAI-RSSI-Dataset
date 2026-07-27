import numpy as np
from scipy import signal

from pipeline_contract import TRADITIONAL_FEATURE_NAMES

def extract_features(orig, faulty, x, split_position=None):
    delta = faulty - orig
    x = np.asarray(x, dtype=float)
    N = len(orig)
    dx = x[1] - x[0] if N > 1 else 1.0
    mb = np.mean(delta)
    rmse = np.sqrt(np.mean(delta**2))
    max_ae = np.max(np.abs(delta))
    window_len = min(20, N // 5)
    local_stds = []
    for i in range(0, N - window_len, window_len // 2):
        local_stds.append(np.std(delta[i:i+window_len]))
    local_std_mean = np.mean(local_stds) if local_stds else 0
    grad_delta = np.gradient(delta, dx)
    grad_grad = np.gradient(grad_delta, dx)
    slope_change_int = np.sum(np.abs(grad_grad)) * dx
    if split_position is None:
        split_position = float(np.median(x))
    left_mask = x <= float(split_position)
    right_mask = x > float(split_position)
    if not np.any(left_mask) or not np.any(right_mask):
        mid_idx = max(1, N // 2)
        left_mask = np.arange(N) < mid_idx
        right_mask = ~left_mask
    mb_left = float(np.mean(delta[left_mask]))
    mb_right = float(np.mean(delta[right_mask]))
    max_abs_mb = max(abs(mb_left), abs(mb_right))
    asymmetry = abs(mb_left - mb_right) / (max_abs_mb + 1e-8)
    f, Pxx = signal.periodogram(delta, fs=1.0/dx, window='hamming', scaling='spectrum')
    valid = f > 0
    f_valid = f[valid]
    Pxx_valid = Pxx[valid]
    if len(Pxx_valid) == 0:
        spr = 1.0
        hfer = 0.0
    else:
        max_peak = np.max(Pxx_valid)
        mean_psd = np.mean(Pxx_valid)
        spr = max_peak / (mean_psd + 1e-8)
        fc = 0.1
        high_freq_mask = f_valid > fc
        total_energy = np.sum(Pxx_valid)
        high_energy = np.sum(Pxx_valid[high_freq_mask])
        hfer = high_energy / (total_energy + 1e-8)
    missing_ratio = np.mean(faulty < -140.0)
    max_slope = 0.5
    diff_faulty = np.diff(faulty) / dx
    asc = np.sum(np.abs(diff_faulty) > max_slope)
    rmse_left = float(np.sqrt(np.mean(delta[left_mask] ** 2)))
    rmse_right = float(np.sqrt(np.mean(delta[right_mask] ** 2)))
    max_abs_left = float(np.max(np.abs(delta[left_mask])))
    max_abs_right = float(np.max(np.abs(delta[right_mask])))
    effect_weights = np.abs(delta)
    effect_centroid_offset = float(
        np.sum(effect_weights * (x - float(split_position)))
        / (np.sum(effect_weights) + 1e-12)
    )
    features = {
        'mb': mb, 'rmse': rmse, 'max_ae': max_ae,
        'local_std_mean': local_std_mean, 'slope_change_int': slope_change_int,
        'asymmetry': asymmetry, 'spr': spr, 'hfer': hfer,
        'missing_ratio': missing_ratio, 'asc': asc,
        'mean_left_dB': mb_left, 'mean_right_dB': mb_right,
        'rmse_left_dB': rmse_left, 'rmse_right_dB': rmse_right,
        'max_abs_left_dB': max_abs_left, 'max_abs_right_dB': max_abs_right,
        'effect_centroid_offset_m': effect_centroid_offset,
    }
    return features

def features_to_array(features_dict):
    return np.array([features_dict[k] for k in TRADITIONAL_FEATURE_NAMES])
