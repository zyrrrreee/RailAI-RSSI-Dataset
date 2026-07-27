import numpy as np


def _split_index(x, split_position=None):
    if split_position is None:
        split_position = float(np.median(x))
    left_mask = x <= split_position
    right_mask = x > split_position
    return split_position, left_mask, right_mask


def _zone_profile(x, center, width_m):
    width_m = max(float(width_m), 1e-6)
    distance = np.abs(x - center)
    mask = distance <= width_m
    profile = np.zeros_like(x, dtype=float)
    profile[mask] = 1.0 - (distance[mask] / width_m) ** 2
    return mask, profile


def fault_global_power_attenuation(rssi, x, atten_dB=6.0):
    return np.asarray(rssi, dtype=float) - float(atten_dB)


def fault_antenna_1_power_loss(rssi, x, drop_dB=7.0, split_position=None):
    split_position, left_mask, _ = _split_index(x, split_position)
    rssi_out = np.asarray(rssi, dtype=float).copy()
    rssi_out[left_mask] -= float(drop_dB)
    return rssi_out


def fault_antenna_2_power_loss(rssi, x, drop_dB=7.0, split_position=None):
    split_position, _, right_mask = _split_index(x, split_position)
    rssi_out = np.asarray(rssi, dtype=float).copy()
    rssi_out[right_mask] -= float(drop_dB)
    return rssi_out


def _tilt_loss_profile(x, split_position, side_mask, tilt_deg):
    """Display-only approximation for an already generated RSSI curve.

    Training data does not use this approximation; it rotates the antenna
    pattern in ``signal_generation.generate_fault_rssi_pair``.
    """
    side_distance = np.abs(np.asarray(x, dtype=float) - float(split_position))
    span = max(float(np.max(side_distance[side_mask])), 1e-6)
    normalized = side_distance / span
    severity = max(abs(float(tilt_deg)), 0.0) / 20.0
    return severity * (1.0 + 8.0 * normalized**1.4)


def fault_antenna_1_tilt(rssi, x, tilt_deg=10.0, split_position=None, tilt_dB_per_m=None):
    split_position, left_mask, _ = _split_index(x, split_position)
    rssi_out = np.asarray(rssi, dtype=float).copy()
    if tilt_dB_per_m is not None:
        tilt_deg = float(tilt_dB_per_m) * 100.0
    loss = _tilt_loss_profile(x, split_position, left_mask, tilt_deg)
    rssi_out[left_mask] -= loss[left_mask]
    return rssi_out


def fault_antenna_2_tilt(rssi, x, tilt_deg=10.0, split_position=None, tilt_dB_per_m=None):
    split_position, _, right_mask = _split_index(x, split_position)
    rssi_out = np.asarray(rssi, dtype=float).copy()
    if tilt_dB_per_m is not None:
        tilt_deg = float(tilt_dB_per_m) * 100.0
    loss = _tilt_loss_profile(x, split_position, right_mask, tilt_deg)
    rssi_out[right_mask] -= loss[right_mask]
    return rssi_out


def fault_feeder_connector_loss(rssi, x, atten_dB=4.0, edge_extra_dB=2.0):
    x = np.asarray(x, dtype=float)
    rssi_out = np.asarray(rssi, dtype=float).copy()
    center = float(np.mean(x))
    span = max(float(np.max(x) - np.min(x)), 1e-6)
    edge_weight = np.abs(x - center) / (span / 2.0)
    edge_weight = np.clip(edge_weight, 0.0, 1.0)
    rssi_out -= float(atten_dB) + float(edge_extra_dB) * edge_weight
    return rssi_out


def fault_onboard_receiver_degradation(rssi, x, atten_dB=3.0, noise_sigma_dB=1.2):
    rssi_out = np.asarray(rssi, dtype=float).copy()
    rssi_out -= float(atten_dB)
    noise = np.random.normal(0.0, float(noise_sigma_dB), len(rssi_out))
    return rssi_out + noise


def fault_local_obstruction(rssi, x, center=None, depth_dB=10.0, width_m=25.0):
    x = np.asarray(x, dtype=float)
    if center is None:
        center = float(np.random.uniform(np.min(x), np.max(x)))
    mask, profile = _zone_profile(x, float(center), float(width_m))
    rssi_out = np.asarray(rssi, dtype=float).copy()
    rssi_out[mask] -= float(depth_dB) * profile[mask]
    return rssi_out


def fault_local_interference(rssi, x, center=None, width_m=30.0, ripple_dB=3.0, noise_sigma_dB=1.0):
    x = np.asarray(x, dtype=float)
    if center is None:
        center = float(np.random.uniform(np.min(x), np.max(x)))
    mask, profile = _zone_profile(x, float(center), float(width_m))
    rssi_out = np.asarray(rssi, dtype=float).copy()
    local_x = x[mask] - float(center)
    if len(local_x) > 0:
        phase = 2.0 * np.pi * local_x / max(float(width_m), 1e-6)
        ripple = float(ripple_dB) * np.sin(phase)
        noise = np.random.normal(0.0, float(noise_sigma_dB), len(local_x))
        rssi_out[mask] += profile[mask] * (ripple + noise)
    return rssi_out


FAULTS = [
    ("全链路功率衰减", fault_global_power_attenuation, {"atten_dB": 8.0}),
    ("天线1功率下降", fault_antenna_1_power_loss, {"drop_dB": 7.0}),
    ("天线2功率下降", fault_antenna_2_power_loss, {"drop_dB": 7.0}),
    ("天线1方向偏移", fault_antenna_1_tilt, {"tilt_deg": 10.0}),
    ("天线2方向偏移", fault_antenna_2_tilt, {"tilt_deg": 10.0}),
]


FAULT_TEMPLATES = [
    ("全链路功率衰减", fault_global_power_attenuation, {"atten_dB": (4.0, 12.0)}),
    ("天线1功率下降", fault_antenna_1_power_loss, {"drop_dB": (3.0, 12.0)}),
    ("天线2功率下降", fault_antenna_2_power_loss, {"drop_dB": (3.0, 12.0)}),
    ("天线1方向偏移", fault_antenna_1_tilt, {"tilt_deg": (2.0, 20.0)}),
    ("天线2方向偏移", fault_antenna_2_tilt, {"tilt_deg": (2.0, 20.0)}),
]


def generate_random_kwargs(param_ranges):
    kwargs = {}
    for pname, prange in param_ranges.items():
        if isinstance(prange, tuple):
            low, high = prange
            if isinstance(low, int) and isinstance(high, int):
                kwargs[pname] = np.random.randint(low, high + 1)
            else:
                kwargs[pname] = np.random.uniform(low, high)
        else:
            kwargs[pname] = prange
    return kwargs
