import unittest

import numpy as np

from signal_generation import (
    STAGE1_OBSERVATION_DEFINITION,
    STAGE1_TOPOLOGY_SCHEMA_VERSION,
    generate_fault_rssi_pair,
    generate_rssi_simulation,
    generate_stage1_obm_observation,
)
from gui_parameter_contract import validate_gui_generator_contract
from pipeline_contract import effective_generator_config
from receiver_measurement import integrate_rssi_dbm_causally, mw_to_dbm


DETERMINISTIC_PARAMS = {
    "x_start": -100.0,
    "x_end": 100.0,
    "dx": 0.5,
    "sigma_shadow": 0.0,
    "K_linear": 1.0e24,
    "receiver_noise_sigma_dB": 0.0,
    "rssi_quantization_dB": 0.0,
    "receiver_sensitivity_dBm": None,
    "trip_power_sigma_dB": 0.0,
    "position_alignment_sigma_m": 0.0,
    "pointing_jitter_sigma_deg": 0.0,
    "measurement_window_m": 0.5,
    "seed": 7,
}


class Stage1RssiModelTests(unittest.TestCase):
    def test_legacy_generator_tuple_remains_compatible(self):
        result = generate_rssi_simulation(**DETERMINISTIC_PARAMS)
        self.assertEqual(len(result), 6)
        self.assertTrue(all(len(item) == len(result[0]) for item in result))

    def test_explicit_topology_and_observation_contract(self):
        observation = generate_stage1_obm_observation(**DETERMINISTIC_PARAMS)
        metadata = observation["metadata"]
        self.assertEqual(
            observation["topology_schema_version"], STAGE1_TOPOLOGY_SCHEMA_VERSION
        )
        self.assertEqual(
            observation["observation_definition"], STAGE1_OBSERVATION_DEFINITION
        )
        self.assertEqual(metadata["ap_count"], 1)
        self.assertEqual(metadata["obm_count"], 1)
        np.testing.assert_allclose(
            metadata["antenna_1_main_lobe_unit_vector"],
            -np.asarray(metadata["antenna_2_main_lobe_unit_vector"]),
        )

    def test_directional_branch_selection_precedes_reporting(self):
        observation = generate_stage1_obm_observation(**DETERMINISTIC_PARAMS)
        x = observation["x_m"]
        selected = observation["selected_antenna"]
        self.assertTrue(np.all(selected[x < 0.0] == 1))
        self.assertTrue(np.all(selected[x > 0.0] == 2))
        expected = np.maximum(
            observation["antenna_1_candidate_raw_rssi_dBm"],
            observation["antenna_2_candidate_raw_rssi_dBm"],
        )
        np.testing.assert_allclose(observation["reported_rssi_dBm"], expected)

    def test_antenna_1_power_fault_changes_only_branch_1_candidate(self):
        _, _, _, metadata = generate_fault_rssi_pair(
            "天线1功率下降",
            {"drop_dB": 8.0},
            return_metadata=True,
            **DETERMINISTIC_PARAMS,
        )
        np.testing.assert_allclose(
            metadata["faulty_antenna_1_candidate_raw_rssi_dBm"],
            metadata["healthy_antenna_1_candidate_raw_rssi_dBm"] - 8.0,
        )
        np.testing.assert_allclose(
            metadata["faulty_antenna_2_candidate_raw_rssi_dBm"],
            metadata["healthy_antenna_2_candidate_raw_rssi_dBm"],
        )

    def test_stage1_api_rejects_single_antenna_scene(self):
        with self.assertRaises(ValueError):
            generate_stage1_obm_observation(dual_antenna=False)

    def test_receiver_floor_saturation_and_quantization_are_enforced(self):
        params = dict(DETERMINISTIC_PARAMS)
        params.update(
            {
                "Pt_dBm": 60.0,
                "receiver_sensitivity_dBm": -99.3,
                "receiver_saturation_dBm": -20.2,
                "rssi_quantization_dB": 0.5,
            }
        )
        reported = generate_rssi_simulation(**params)[1]
        self.assertGreaterEqual(float(np.min(reported)), -99.3)
        self.assertLessEqual(float(np.max(reported)), -20.2)

    def test_sample_semantics_are_recorded(self):
        metadata = generate_rssi_simulation(
            return_metadata=True, **DETERMINISTIC_PARAMS
        )[6]
        self.assertEqual(
            metadata["position_grid_semantics"],
            "OBM_position_aligned_and_resampled_to_fixed_spatial_grid",
        )
        self.assertEqual(metadata["independent_simulation_unit"], "one_complete_trip_identified_by_seed")
        self.assertAlmostEqual(metadata["spatial_sample_interval_m"], 0.5)

    def test_deprecated_path_count_does_not_change_generator_contract(self):
        default_config = effective_generator_config(None)
        legacy_config = effective_generator_config({"N_paths": 999})
        self.assertEqual(default_config, legacy_config)
        self.assertNotIn("N_paths", default_config)
        self.assertEqual(validate_gui_generator_contract(), [])

    def test_ap_obm_distance_uses_three_dimensional_geometry(self):
        params = dict(DETERMINISTIC_PARAMS)
        params.update(
            {
                "ap_height_m": 8.0,
                "obm_height_m": 3.0,
                "antenna_y": 5.0,
            }
        )
        result = generate_rssi_simulation(return_metadata=True, **params)
        center = int(np.argmin(np.abs(result[0])))
        self.assertAlmostEqual(result[5][center], np.sqrt(5.0**2 + 5.0**2))
        self.assertEqual(
            result[6]["geometry_model"],
            "3D_distance_with_horizontal_azimuth_antenna_pattern",
        )

    def test_receiver_integrates_complete_candidate_link_in_linear_power(self):
        params = dict(DETERMINISTIC_PARAMS)
        params.update({"K_linear": 0.0, "measurement_window_m": 1.5})
        metadata = generate_rssi_simulation(return_metadata=True, **params)[6]
        instantaneous = np.asarray(
            metadata["antenna_1_candidate_instantaneous_rssi_dBm"], dtype=float
        )
        coordinate = np.arange(len(instantaneous), dtype=float) * 0.5
        integration = integrate_rssi_dbm_causally(
            instantaneous,
            coordinate,
            1.5,
        )
        expected = mw_to_dbm(integration.averaged_power_mW)
        np.testing.assert_allclose(
            metadata["antenna_1_candidate_raw_rssi_dBm"], expected
        )
        self.assertEqual(metadata["measurement_window_samples"], 3)
        self.assertEqual(
            metadata["receiver_integration_alignment"], "causal_trailing_window"
        )
        self.assertEqual(metadata["receiver_integration_coordinate_unit"], "m")

    def test_optional_nonstationary_profile_keeps_stage1_topology(self):
        params = dict(DETERMINISTIC_PARAMS)
        params.update(
            {
                "path_loss_breakpoint_m": 50.0,
                "path_loss_exponent_far": 3.88,
                "shadow_sigma_far_dB": 4.2,
                "rician_K_slope_dB_per_100m": -1.0,
            }
        )
        metadata = generate_rssi_simulation(return_metadata=True, **params)[6]
        self.assertEqual(metadata["ap_count"], 1)
        self.assertEqual(metadata["obm_count"], 1)
        self.assertEqual(
            metadata["scene"],
            "single_wayside_ap_dual_opposite_directional_antennas_single_onboard_obm",
        )
        np.testing.assert_allclose(
            metadata["antenna_1_main_lobe_unit_vector"],
            -np.asarray(metadata["antenna_2_main_lobe_unit_vector"]),
        )
        self.assertEqual(metadata["path_loss_model"], "continuous_two_slope_log_distance")
        self.assertGreater(
            float(np.ptp(np.asarray(metadata["rician_K_profile_dB"], dtype=float))),
            0.0,
        )

    def test_free_space_mode_is_not_reported_as_two_slope_path_loss(self):
        params = dict(DETERMINISTIC_PARAMS)
        params.update({"use_free_space": True, "path_loss_breakpoint_m": 50.0})
        metadata = generate_rssi_simulation(return_metadata=True, **params)[6]
        self.assertEqual(metadata["path_loss_model"], "free_space")


if __name__ == "__main__":
    unittest.main()
