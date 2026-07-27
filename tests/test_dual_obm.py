import unittest

import numpy as np

from dual_obm import generate_dual_obm_observation
from signal_generation import TIME_DOMAIN_SAMPLING


DETERMINISTIC_COMMON = {
    "x_start": -400.0,
    "x_end": 150.0,
    "v": 20.0,
    "sampling_mode": TIME_DOMAIN_SAMPLING,
    "simulation_step_s": 0.01,
    "report_interval_s": 0.05,
    "seed": 123,
    "position_alignment_sigma_m": 0.0,
    "pointing_jitter_sigma_deg": 0.0,
    "trip_power_sigma_dB": 0.0,
}


class DualObmTests(unittest.TestCase):
    def test_front_and_rear_are_separated_by_train_length_forward(self):
        result = generate_dual_obm_observation(
            train_length_m=200.0, **DETERMINISTIC_COMMON
        )
        front = result["front_obm"]["time_domain"]["x_m"]
        rear = result["rear_obm"]["time_domain"]["x_m"]
        centre = result["train_center_position_m"]
        self.assertLess(len(centre), len(front))
        np.testing.assert_allclose(front - rear, 200.0, atol=1e-10)
        np.testing.assert_allclose(
            0.5 * (result["front_obm"]["x_m"] + result["rear_obm"]["x_m"]),
            centre,
            atol=1e-10,
        )
        self.assertEqual(
            result["metadata"]["train_reference_position"], "geometric_centre"
        )

    def test_front_definition_follows_reverse_motion_not_coordinate_size(self):
        result = generate_dual_obm_observation(
            train_length_m=120.0,
            x_start=150.0,
            x_end=-400.0,
            v=20.0,
            sampling_mode=TIME_DOMAIN_SAMPLING,
            simulation_step_s=0.01,
            report_interval_s=0.05,
            seed=123,
            position_alignment_sigma_m=0.0,
            pointing_jitter_sigma_deg=0.0,
            trip_power_sigma_dB=0.0,
        )
        front = result["front_obm"]["time_domain"]["x_m"]
        rear = result["rear_obm"]["time_domain"]["x_m"]
        self.assertTrue(np.all(front < rear))
        np.testing.assert_allclose(rear - front, 120.0, atol=1e-10)
        self.assertEqual(result["metadata"]["motion_direction"], -1)

    def test_constant_speed_crossing_delay_is_length_over_speed(self):
        for speed in (10.0, 20.0, 30.0):
            with self.subTest(speed=speed):
                params = dict(DETERMINISTIC_COMMON)
                params["v"] = speed
                result = generate_dual_obm_observation(
                    train_length_m=150.0, **params
                )
                metadata = result["metadata"]
                self.assertAlmostEqual(
                    metadata["actual_front_to_rear_crossing_delay_s"],
                    150.0 / speed,
                    places=8,
                )
                self.assertAlmostEqual(
                    metadata["expected_constant_speed_crossing_delay_s"],
                    150.0 / speed,
                    places=12,
                )

    def test_shadow_is_one_spatial_field_but_small_fading_is_independent(self):
        result = generate_dual_obm_observation(
            train_length_m=100.0, **DETERMINISTIC_COMMON
        )
        front = result["front_obm"]["time_domain"]
        rear = result["rear_obm"]["time_domain"]
        overlap_min = max(float(np.min(front["x_m"])), float(np.min(rear["x_m"])))
        overlap_max = min(float(np.max(front["x_m"])), float(np.max(rear["x_m"])))
        front_mask = (front["x_m"] >= overlap_min) & (front["x_m"] <= overlap_max)
        rear_order = np.argsort(rear["x_m"])
        rear_shadow_at_front = np.interp(
            front["x_m"][front_mask],
            rear["x_m"][rear_order],
            rear["shared_shadow_dB"][rear_order],
        )
        np.testing.assert_allclose(
            front["shared_shadow_dB"][front_mask],
            rear_shadow_at_front,
            atol=1e-8,
        )
        self.assertFalse(
            np.allclose(
                front["independent_small_fading_dB"],
                rear["independent_small_fading_dB"],
            )
        )

    def test_receiver_noise_is_independent_and_whole_run_is_reproducible(self):
        first = generate_dual_obm_observation(
            train_length_m=200.0, **DETERMINISTIC_COMMON
        )
        repeated = generate_dual_obm_observation(
            train_length_m=200.0, **DETERMINISTIC_COMMON
        )
        np.testing.assert_array_equal(
            first["front_obm"]["reported_rssi_dBm"],
            repeated["front_obm"]["reported_rssi_dBm"],
        )
        np.testing.assert_array_equal(
            first["rear_obm"]["reported_rssi_dBm"],
            repeated["rear_obm"]["reported_rssi_dBm"],
        )
        self.assertFalse(
            np.array_equal(
                first["front_obm"]["time_domain"][
                    "independent_receiver_noise_dB"
                ],
                first["rear_obm"]["time_domain"][
                    "independent_receiver_noise_dB"
                ],
            )
        )

    def test_two_obm_reports_are_not_merged_or_selected(self):
        result = generate_dual_obm_observation(
            train_length_m=200.0, **DETERMINISTIC_COMMON
        )
        self.assertIn("front_obm", result)
        self.assertIn("rear_obm", result)
        self.assertNotIn("combined_rssi_dBm", result)
        self.assertNotIn("selected_obm", result)
        self.assertEqual(
            result["metadata"]["obm_combining_policy"],
            "none_keep_independent_traces",
        )

    def test_report_grids_are_common_in_time_and_distinct_in_position(self):
        result = generate_dual_obm_observation(
            train_length_m=200.0, **DETERMINISTIC_COMMON
        )
        front = result["front_obm"]
        rear = result["rear_obm"]
        np.testing.assert_array_equal(front["t_s"], rear["t_s"])
        np.testing.assert_allclose(front["x_m"] - rear["x_m"], 200.0)
        self.assertEqual(len(front["reported_rssi_dBm"]), 551)
        self.assertEqual(len(rear["reported_rssi_dBm"]), 551)

    def test_invalid_length_and_legacy_sampling_are_rejected(self):
        with self.assertRaises(ValueError):
            generate_dual_obm_observation(
                train_length_m=0.0, **DETERMINISTIC_COMMON
            )
        params = dict(DETERMINISTIC_COMMON)
        params["sampling_mode"] = "fixed_space_legacy"
        with self.assertRaises(ValueError):
            generate_dual_obm_observation(train_length_m=200.0, **params)


if __name__ == "__main__":
    unittest.main()
