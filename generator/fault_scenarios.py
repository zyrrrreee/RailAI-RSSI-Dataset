"""Trace-level channel profiles and physically interpretable fault scenarios.

The ranges are deliberately broad priors for simulation experiments, not
claims about one target railway.  They remain inside the measurement-informed
intervals used by ``simulation_validation.py`` and must later be calibrated
with aligned healthy runs from the target line.
"""

from __future__ import annotations

from typing import Any

import numpy as np


HEALTHY_LABEL = "健康"
COMPOSITE_LABEL = "复合故障"
SEVERITY_LEVELS = ("轻微", "中等", "严重")


FAULT_COMPONENT_COLUMNS = {
    "全链路功率衰减": "has_global_attenuation",
    "天线1功率下降": "has_antenna_1_power_loss",
    "天线2功率下降": "has_antenna_2_power_loss",
    "天线1方向偏移": "has_antenna_1_tilt",
    "天线2方向偏移": "has_antenna_2_tilt",
}


FAULT_SCENARIO_SPECS = {
    "全链路功率衰减": {
        "parameter": "atten_dB",
        "component": "发射机、馈线或公共射频链路",
        "mechanism": "公共链路预算下降，双天线覆盖区整体受影响",
        "ranges": {
            "轻微": (1.5, 4.0),
            "中等": (4.0, 8.0),
            "严重": (8.0, 12.0),
        },
    },
    "天线1功率下降": {
        "parameter": "drop_dB",
        "component": "天线1支路功率或馈线效率",
        "mechanism": "天线1支路增益下降，并允许天线2支路接管",
        "ranges": {
            "轻微": (1.5, 3.5),
            "中等": (3.5, 7.0),
            "严重": (7.0, 12.0),
        },
    },
    "天线2功率下降": {
        "parameter": "drop_dB",
        "component": "天线2支路功率或馈线效率",
        "mechanism": "天线2支路增益下降，并允许天线1支路接管",
        "ranges": {
            "轻微": (1.5, 3.5),
            "中等": (3.5, 7.0),
            "严重": (7.0, 12.0),
        },
    },
    "天线1方向偏移": {
        "parameter": "tilt_deg",
        "component": "天线1安装方向或机械姿态",
        "mechanism": "重新旋转天线1方向图，形成随位置变化的覆盖损失",
        "ranges": {
            "轻微": (2.0, 5.0),
            "中等": (5.0, 10.0),
            "严重": (10.0, 18.0),
        },
    },
    "天线2方向偏移": {
        "parameter": "tilt_deg",
        "component": "天线2安装方向或机械姿态",
        "mechanism": "重新旋转天线2方向图，形成随位置变化的覆盖损失",
        "ranges": {
            "轻微": (2.0, 5.0),
            "中等": (5.0, 10.0),
            "严重": (10.0, 18.0),
        },
    },
}


COMPOSITE_TEMPLATES = (
    ("全链路功率衰减", "天线1方向偏移"),
    ("全链路功率衰减", "天线2方向偏移"),
    ("天线1功率下降", "天线1方向偏移"),
    ("天线2功率下降", "天线2方向偏移"),
    ("天线1功率下降", "天线2功率下降"),
)


CHANNEL_PROFILE_SPECS = {
    "rail_los_low_variability": {
        "description": "LOS 条件较稳定的低波动先验",
        "ranges": {
            "n": (2.2, 2.7),
            "sigma_shadow": (2.0, 3.0),
            "shadow_corr_distance_m": (15.0, 35.0),
            "rician_K_dB": (5.0, 9.0),
            "PL0_dB": (38.0, 42.0),
            "Pt_dBm": (18.0, 22.0),
            "antenna_y": (4.0, 7.0),
            "theta_half_deg": (26.0, 36.0),
            "receiver_noise_sigma_dB": (0.35, 0.70),
            "trip_power_sigma_dB": (0.40, 0.80),
            "position_alignment_sigma_m": (0.25, 0.75),
            "pointing_jitter_sigma_deg": (0.50, 1.50),
            "v": (14.0, 30.0),
        },
    },
    "rail_los_nominal": {
        "description": "与当前公开铁路测量先验接近的名义 LOS 条件",
        "ranges": {
            "n": (2.6, 3.0),
            "sigma_shadow": (2.5, 4.0),
            "shadow_corr_distance_m": (10.0, 25.0),
            "rician_K_dB": (3.0, 7.0),
            "PL0_dB": (39.0, 43.0),
            "Pt_dBm": (18.0, 22.0),
            "antenna_y": (4.0, 8.0),
            "theta_half_deg": (24.0, 36.0),
            "receiver_noise_sigma_dB": (0.45, 0.90),
            "trip_power_sigma_dB": (0.60, 1.00),
            "position_alignment_sigma_m": (0.50, 1.20),
            "pointing_jitter_sigma_deg": (0.75, 2.00),
            "v": (12.0, 32.0),
        },
    },
    "rail_los_high_variability": {
        "description": "仍在文献先验内的高阴影和多径波动条件",
        "ranges": {
            "n": (2.9, 3.4),
            "sigma_shadow": (4.0, 5.8),
            "shadow_corr_distance_m": (6.0, 18.0),
            "rician_K_dB": (0.5, 4.0),
            "PL0_dB": (40.0, 44.0),
            "Pt_dBm": (18.0, 22.0),
            "antenna_y": (4.0, 8.0),
            "theta_half_deg": (22.0, 34.0),
            "receiver_noise_sigma_dB": (0.70, 1.20),
            "trip_power_sigma_dB": (0.80, 1.30),
            "position_alignment_sigma_m": (0.75, 1.75),
            "pointing_jitter_sigma_deg": (1.25, 2.75),
            "v": (10.0, 35.0),
        },
    },
}


def severity_for_index(index: int) -> str:
    return SEVERITY_LEVELS[int(index) % len(SEVERITY_LEVELS)]


def _sample_range(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return float(rng.uniform(float(low), float(high)))


def sample_channel_profile(
    profile_name: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if profile_name not in CHANNEL_PROFILE_SPECS:
        raise ValueError(f"未知信道 profile: {profile_name}")
    spec = CHANNEL_PROFILE_SPECS[profile_name]
    sampled = {
        key: _sample_range(rng, bounds)
        for key, bounds in spec["ranges"].items()
    }
    rician_k_db = sampled.pop("rician_K_dB")
    sampled["K_linear"] = 10.0 ** (rician_k_db / 10.0)
    sampled["measurement_window_m"] = None
    sampled["measurement_window_s"] = 0.05
    return {
        "profile_name": profile_name,
        "profile_description": spec["description"],
        "rician_K_dB": float(rician_k_db),
        "generator_overrides": sampled,
    }


def sample_fault_scenario(
    fault_type: str,
    severity_level: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if fault_type not in FAULT_SCENARIO_SPECS:
        raise ValueError(f"未知单故障类型: {fault_type}")
    if severity_level not in SEVERITY_LEVELS:
        raise ValueError(f"未知严重度: {severity_level}")
    spec = FAULT_SCENARIO_SPECS[fault_type]
    parameter = spec["parameter"]
    value = _sample_range(rng, spec["ranges"][severity_level])
    global_low = float(spec["ranges"][SEVERITY_LEVELS[0]][0])
    global_high = float(spec["ranges"][SEVERITY_LEVELS[-1]][1])
    score = (value - global_low) / max(global_high - global_low, 1e-12)
    return {
        "fault_type": fault_type,
        "fault_components": [fault_type],
        "fault_kwargs": {parameter: value},
        "severity_level": severity_level,
        "severity_rank": SEVERITY_LEVELS.index(severity_level) + 1,
        "severity_score": float(np.clip(score, 0.0, 1.0)),
        "component": spec["component"],
        "mechanism": spec["mechanism"],
        "temporal_behavior": "persistent_during_trip",
    }


def sample_composite_scenario(
    template_index: int,
    severity_level: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    component_names = COMPOSITE_TEMPLATES[int(template_index) % len(COMPOSITE_TEMPLATES)]
    components = [
        sample_fault_scenario(name, severity_level, rng) for name in component_names
    ]
    return {
        "fault_type": COMPOSITE_LABEL,
        "fault_components": list(component_names),
        "component_specs": [
            (item["fault_type"], item["fault_kwargs"]) for item in components
        ],
        "severity_level": severity_level,
        "severity_rank": SEVERITY_LEVELS.index(severity_level) + 1,
        "severity_score": float(np.mean([item["severity_score"] for item in components])),
        "component": " + ".join(item["component"] for item in components),
        "mechanism": " + ".join(item["mechanism"] for item in components),
        "temporal_behavior": "persistent_during_trip",
    }


def component_indicator_values(component_names: list[str]) -> dict[str, int]:
    selected = set(component_names)
    return {
        column: int(fault_name in selected)
        for fault_name, column in FAULT_COMPONENT_COLUMNS.items()
    }
