import unittest

import numpy as np

from motion_sampling import generate_trajectory
from signal_generation import (
    LEGACY_SPATIAL_SAMPLING,
    TIME_DOMAIN_SAMPLING,
    generate_fault_rssi_pair,
    generate_ideal_rssi,
    generate_rssi_simulation,
    generate_time_domain_rssi_simulation,
)


class TimePositionSamplingTests(unittest.TestCase):
    def test_fixed_report_period_changes_spatial_spacing_with_speed(self):
        expected = {10.0: 0.5, 20.0: 1.0, 30.0: 1.5}
        expected_duration = {10.0: 55.0, 20.0: 27.5, 30.0: 550.0 / 30.0}
        for speed, spacing in expected.items():
            with self.subTest(speed=speed):
                trajectory = generate_trajectory(
                    -400.0,
                    150.0,
                    simulation_step_s=0.01,
                    report_interval_s=0.05,
                    speed_mps=speed,
                )
                regular_steps = np.abs(np.diff(trajectory.report_x_m))[:-1]
                self.assertTrue(np.allclose(regular_steps, spacing, atol=1e-10))
                self.assertAlmostEqual(
                    trajectory.report_t_s[-1], expected_duration[speed], places=10
                )

    def test_same_position_ideal_link_budget_is_speed_invariant(self):
        common = {
            "sampling_mode": TIME_DOMAIN_SAMPLING,
            "simulation_step_s": 0.01,
            "report_interval_s": 0.05,
        }
        slow = generate_ideal_rssi(v=10.0, **common)
        fast = generate_ideal_rssi(v=30.0, **common)
        slow_positions = np.round(slow[0], 8)
        fast_positions = np.round(fast[0], 8)
        _, slow_index, fast_index = np.intersect1d(
            slow_positions, fast_positions, return_indices=True
        )
        self.assertGreater(len(slow_index), 300)
        np.testing.assert_allclose(slow[2][slow_index], fast[2][fast_index], atol=1e-12)
        np.testing.assert_allclose(slow[3][slow_index], fast[3][fast_index], atol=1e-12)
        np.testing.assert_allclose(slow[1][slow_index], fast[1][fast_index], atol=1e-12)

    def test_piecewise_speed_profile_is_monotonic_continuous_and_bounded(self):
        trajectory = generate_trajectory(
            0.0,
            150.0,
            simulation_step_s=0.02,
            report_interval_s=0.1,
            speed_mps=None,
            speed_profile=[(0.0, 5.0), (5.0, 15.0), (10.0, 10.0)],
        )
        self.assertTrue(np.all(np.diff(trajectory.t_s) > 0.0))
        self.assertTrue(np.all(np.diff(trajectory.x_m) > 0.0))
        self.assertLessEqual(float(np.max(trajectory.x_m)), 150.0)
        self.assertAlmostEqual(float(trajectory.x_m[-1]), 150.0, places=10)
        self.assertAlmostEqual(float(trajectory.t_s[-1]), 13.75, places=8)
        index_at_five = int(np.argmin(np.abs(trajectory.t_s - 5.0)))
        self.assertAlmostEqual(float(trajectory.v_mps[index_at_five]), 15.0, places=10)

    def test_trajectory_is_reproducible_and_supports_reverse_motion(self):
        kwargs = dict(
            x_start=100.0,
            x_end=0.0,
            simulation_step_s=0.01,
            report_interval_s=0.05,
            speed_mps=20.0,
            direction=-1,
        )
        first = generate_trajectory(**kwargs)
        second = generate_trajectory(**kwargs)
        np.testing.assert_array_equal(first.t_s, second.t_s)
        np.testing.assert_array_equal(first.x_m, second.x_m)
        np.testing.assert_array_equal(first.v_mps, second.v_mps)
        self.assertTrue(np.all(first.v_mps < 0.0))
        self.assertAlmostEqual(float(first.t_s[-1]), 5.0)

    def test_new_time_api_retains_internal_and_report_domain_data(self):
        result = generate_time_domain_rssi_simulation(
            v=20.0,
            seed=123,
            return_metadata=True,
            simulation_step_s=0.01,
            report_interval_s=0.05,
        )
        x, rssi, _, _, _, _, metadata = result
        self.assertEqual(len(x), 551)
        self.assertEqual(len(rssi), len(metadata["report_time_s"]))
        self.assertEqual(metadata["sampling_mode"], TIME_DOMAIN_SAMPLING)
        self.assertEqual(metadata["measurement_window_samples"], 5)
        self.assertEqual(len(metadata["time_domain"]["x_m"]), 2751)
        self.assertEqual(
            len(metadata["time_domain"]["reported_rssi_dBm"]), 2751
        )
        self.assertEqual(
            len(
                metadata["time_domain"][
                    "antenna_1_candidate_instantaneous_rssi_dBm"
                ]
            ),
            2751,
        )
        self.assertEqual(
            len(metadata["antenna_1_candidate_raw_rssi_dBm"]), 551
        )

    def test_legacy_six_array_api_remains_available(self):
        result = generate_rssi_simulation(
            sampling_mode=LEGACY_SPATIAL_SAMPLING,
            x_start=-400.0,
            x_end=150.0,
            dx=0.5,
            seed=123,
        )
        self.assertEqual(len(result), 6)
        self.assertEqual(len(result[0]), 1101)

    def test_time_domain_fault_pair_uses_one_shared_trajectory_and_channel(self):
        x, healthy, faulty, metadata = generate_fault_rssi_pair(
            "\u5168\u94fe\u8def\u529f\u7387\u8870\u51cf",
            {"atten_dB": 8.0},
            sampling_mode=TIME_DOMAIN_SAMPLING,
            simulation_step_s=0.01,
            report_interval_s=0.05,
            seed=123,
            return_metadata=True,
        )
        self.assertEqual(len(x), len(healthy))
        self.assertEqual(len(x), len(faulty))
        self.assertEqual(len(x), len(metadata["report_time_s"]))
        self.assertGreater(float(np.mean(healthy - faulty)), 6.0)
        self.assertEqual(metadata["sampling_mode"], TIME_DOMAIN_SAMPLING)


if __name__ == "__main__":
    unittest.main()
