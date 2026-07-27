"""Single source of truth for GUI-visible RSSI generator parameters."""

from __future__ import annotations

import inspect
from typing import Any, Mapping

import numpy as np

from signal_generation import (
    DEFAULT_RICIAN_K_DB,
    TIME_DOMAIN_SAMPLING,
    generate_ideal_rssi,
    generate_rssi_simulation,
)


BASIC_GENERATOR_PARAMETER_SPECS = (
    ("x_start", "起始位置 (m)", -400.0, -1000.0, 1000.0, 0.5, False),
    ("x_end", "终点位置 (m)", 150.0, -1000.0, 1000.0, 0.5, False),
    ("dx", "旧空间模式步长 (m，时间域不使用)", 0.5, 0.05, 10.0, 0.05, False),
    ("v", "列车速度 (m/s)", 20.0, 0.0, 100.0, 0.5, False),
    ("antenna_x", "轨旁 AP 轨道位置 (m)", 0.0, -500.0, 500.0, 0.5, False),
    ("antenna_y", "轨旁 AP 横向偏移 (m)", 5.0, 0.1, 100.0, 0.5, False),
    ("ap_height_m", "轨旁 AP 天线高度 (m，待标定)", 5.0, 0.0, 50.0, 0.1, False),
    ("obm_height_m", "车载 OBM 天线高度 (m，文献先验)", 4.1, 0.0, 10.0, 0.1, False),
    ("G_max_dB", "AP 单副定向天线最大增益 (dBi)", 12.0, -10.0, 40.0, 0.5, False),
    ("theta_half_deg", "AP 天线半功率角 (°)", 30.0, 1.0, 180.0, 0.5, False),
    ("Pt_dBm", "AP 下行发射功率 (dBm)", 20.0, -50.0, 50.0, 0.5, False),
    ("Gr_dBi", "车载 OBM 接收增益 (dBi)", 0.0, -30.0, 40.0, 0.5, False),
    ("n", "路径损耗指数", 2.8, 1.0, 6.0, 0.05, False),
    ("fc", "载波频率 (Hz)", 2.4e9, 1.0e6, 1.0e12, 1.0e8, False),
    ("sigma_shadow", "阴影衰落标准差 (dB)", 2.5, 0.0, 20.0, 0.1, False),
    ("K_dB", "莱斯 K 因子 (dB)", DEFAULT_RICIAN_K_DB, -20.0, 40.0, 0.1, False),
    ("seed", "随机种子", 123, 0, 2_147_483_646, 1, True),
)


ADVANCED_GENERATOR_PARAMETER_SPECS = (
    ("simulation_step_s", "内部运动/信道步长 (s)", 0.01, 0.001, 1.0, 0.001, False),
    ("report_interval_s", "OBM RSSI报告周期 (s)", 0.05, 0.001, 10.0, 0.01, False),
    ("PL0_dB", "参考距离路径损耗 PL0 (dB)", 40.0, 0.0, 200.0, 0.5, False),
    ("d0", "参考距离 d0 (m)", 1.0, 0.01, 100.0, 0.1, False),
    ("shadow_corr_distance_m", "阴影去相关距离 (m)", 15.0, 0.0, 500.0, 0.5, False),
    ("measurement_window_s", "接收机因果积分时间 (s)", 0.05, 0.0, 10.0, 0.01, False),
    ("receiver_noise_sigma_dB", "接收机噪声标准差 (dB)", 0.6, 0.0, 20.0, 0.05, False),
    ("rssi_quantization_dB", "RSSI 量化步长 (dB)", 0.5, 0.0, 10.0, 0.1, False),
    ("receiver_sensitivity_dBm", "接收机灵敏度 (dBm)", -100.0, -200.0, 0.0, 1.0, False),
    ("receiver_saturation_dBm", "接收机强信号饱和上限 (dBm)", -20.0, -120.0, 50.0, 1.0, False),
    ("max_antenna_attenuation_dB", "方向图最大衰减 (dB)", 30.0, 0.0, 100.0, 1.0, False),
    ("trip_power_sigma_dB", "行程功率偏差标准差 (dB)", 0.8, 0.0, 20.0, 0.1, False),
    ("position_alignment_sigma_m", "位置对齐误差标准差 (m)", 0.75, 0.0, 20.0, 0.05, False),
    ("pointing_jitter_sigma_deg", "天线指向抖动标准差 (°)", 1.5, 0.0, 30.0, 0.1, False),
    ("path_loss_breakpoint_m", "分段路径损耗断点 (m，0=关闭)", 0.0, 0.0, 5000.0, 10.0, False),
    ("path_loss_exponent_far", "断点后路径损耗指数", 3.88, 1.0, 8.0, 0.05, False),
    ("shadow_sigma_far_dB", "断点后阴影标准差 (dB)", 4.2, 0.0, 20.0, 0.1, False),
    ("rician_K_slope_dB_per_100m", "K因子距离斜率 (dB/100m)", 0.0, -10.0, 10.0, 0.1, False),
)


FIXED_GENERATOR_PARAMETERS = {
    "main_lobe_dir_x": -1.0,
    "main_lobe_dir_y": 0.0,
    "use_free_space": False,
    "measurement_window_m": None,
    "below_sensitivity_policy": "clamp_report_preserve_raw_status",
    "dual_antenna": True,
    "sampling_mode": TIME_DOMAIN_SAMPLING,
    "speed_profile": None,
    "direction": None,
}


DEPRECATED_COMPATIBILITY_PARAMETERS = {"N_paths"}
INTENTIONAL_GUI_DEFAULT_OVERRIDES = {"sampling_mode"}


def generator_params_from_gui_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Convert GUI units into the exact public simulator argument contract."""
    result = dict(values)
    missing = {
        item[0]
        for item in BASIC_GENERATOR_PARAMETER_SPECS + ADVANCED_GENERATOR_PARAMETER_SPECS
    } - set(result)
    if missing:
        raise ValueError(f"GUI 缺少仿真参数: {sorted(missing)}")
    if float(result["x_end"]) <= float(result["x_start"]):
        raise ValueError("终点位置必须大于起始位置")
    if float(result["dx"]) <= 0.0:
        raise ValueError("空间采样步长必须大于 0")
    if float(result["simulation_step_s"]) <= 0.0:
        raise ValueError("内部运动/信道步长必须大于 0")
    if float(result["report_interval_s"]) <= 0.0:
        raise ValueError("OBM RSSI报告周期必须大于 0")
    if float(result["simulation_step_s"]) > float(result["report_interval_s"]):
        raise ValueError("内部运动/信道步长不能大于 OBM RSSI报告周期")
    if float(result["receiver_saturation_dBm"]) <= float(
        result["receiver_sensitivity_dBm"]
    ):
        raise ValueError("接收机强信号饱和上限必须高于灵敏度下限")
    k_db = float(result.pop("K_dB"))
    result["K_linear"] = 10.0 ** (k_db / 10.0)
    result.update(FIXED_GENERATOR_PARAMETERS)
    return result


def default_gui_generator_params(*, include_seed: bool = False) -> dict[str, Any]:
    values = {
        key: default
        for key, _label, default, _minimum, _maximum, _step, _is_int in (
            BASIC_GENERATOR_PARAMETER_SPECS + ADVANCED_GENERATOR_PARAMETER_SPECS
        )
    }
    params = generator_params_from_gui_values(values)
    if not include_seed:
        params.pop("seed", None)
    return params


def ideal_generator_params(generator_params: Mapping[str, Any]) -> dict[str, Any]:
    accepted = set(inspect.signature(generate_ideal_rssi).parameters)
    return {key: value for key, value in generator_params.items() if key in accepted}


def validate_gui_generator_contract() -> list[str]:
    """Return every visible/fixed default that diverges from simulator code."""
    errors: list[str] = []
    signature = inspect.signature(generate_rssi_simulation)
    configured = default_gui_generator_params(include_seed=True)
    accounted = (
        set(configured) | {"return_metadata"} | DEPRECATED_COMPATIBILITY_PARAMETERS
    )
    for name, parameter in signature.parameters.items():
        if name not in accounted:
            errors.append(f"仿真参数未被 GUI 或固定契约覆盖: {name}")
            continue
        if name in {"return_metadata", "seed"} | DEPRECATED_COMPATIBILITY_PARAMETERS:
            continue
        if name in INTENTIONAL_GUI_DEFAULT_OVERRIDES:
            continue
        expected = parameter.default
        actual = configured[name]
        if expected is None or actual is None:
            equal = expected is actual
        elif isinstance(expected, bool):
            equal = bool(actual) is bool(expected)
        elif isinstance(expected, str):
            equal = str(actual) == expected
        else:
            equal = bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12))
        if not equal:
            errors.append(f"{name} 默认值不一致: GUI={actual!r}, 仿真器={expected!r}")
    return errors
