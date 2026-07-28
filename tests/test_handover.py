import unittest

import numpy as np

from handover import (
    HandoverConfig,
    causal_moving_average,
    generate_dual_obm_handover,
    normalise_handover_config,
    run_obm_handover,
    time_aware_ewma,
)


def run_two_ap(
    first,
    second,
    *,
    dt=0.1,
    available=None,
    config=None,
):
    reports = np.vstack(
        [np.asarray(first, dtype=float), np.asarray(second, dtype=float)]
    )
    sample_count = reports.shape[1]
    if available is None:
        available = np.ones(reports.shape, dtype=bool)
    return run_obm_handover(
        time_s=np.arange(sample_count, dtype=float) * float(dt),
        position_m=np.arange(sample_count, dtype=float) * 2.0,
        report_matrix_dBm=reports,
        available_matrix=np.asarray(available, dtype=bool),
        ap_ids=("AP-A", "AP-B"),
        obm_id="test-obm",
        config=config,
    )


class StatefulHandoverTests(unittest.TestCase):
    def test_causal_moving_average_never_uses_future_samples(self):
        time = np.array([0.0, 0.1, 0.2, 0.3])
        original = np.array([-60.0, -60.0, -60.0, -20.0])
        changed_future = original.copy()
        changed_future[-1] = 20.0
        first = causal_moving_average(
            time,
            original,
            window_s=0.25,
            reset_after_s=1.0,
        )
        second = causal_moving_average(
            time,
            changed_future,
            window_s=0.25,
            reset_after_s=1.0,
        )
        np.testing.assert_array_equal(first[:-1], second[:-1])
        self.assertAlmostEqual(first[2], -60.0)

    def test_time_aware_ewma_uses_elapsed_time(self):
        time = np.array([0.0, 0.1, 0.3])
        values = np.array([-60.0, -50.0, -40.0])
        filtered = time_aware_ewma(
            time,
            values,
            tau_s=0.2,
            reset_after_s=1.0,
        )
        first_alpha = 1.0 - np.exp(-0.1 / 0.2)
        second_alpha = 1.0 - np.exp(-0.2 / 0.2)
        expected_second = (1.0 - first_alpha) * -60.0 + first_alpha * -50.0
        expected_third = (
            (1.0 - second_alpha) * expected_second
            + second_alpha * -40.0
        )
        np.testing.assert_allclose(
            filtered,
            np.array([-60.0, expected_second, expected_third]),
        )

    def test_missing_report_is_nan_and_long_gap_resets_filter(self):
        time = np.array([0.0, 0.1, 1.0])
        values = np.array([-60.0, -40.0, -30.0])
        available = np.array([True, False, True])
        filtered = time_aware_ewma(
            time,
            values,
            available,
            tau_s=0.2,
            reset_after_s=0.5,
        )
        self.assertEqual(filtered[0], -60.0)
        self.assertTrue(np.isnan(filtered[1]))
        self.assertEqual(filtered[2], -30.0)

    def test_initial_association_is_not_counted_as_handover(self):
        result = run_two_ap(
            [-45.0] * 6,
            [-60.0] * 6,
            config=HandoverConfig(decision_filter="none"),
        )
        self.assertEqual(result["summary"]["initial_association_count"], 1)
        self.assertEqual(result["summary"]["handover_completion_count"], 0)
        self.assertTrue(np.all(result["serving_ap_id"] == "AP-A"))
        self.assertEqual(result["events"][0]["event_type"], "initial_association")

    def test_hysteresis_blocks_small_candidate_advantage(self):
        result = run_two_ap(
            [-45.0] + [-50.0] * 7,
            [-60.0] + [-48.0] * 7,
            config=HandoverConfig(
                decision_filter="none",
                hysteresis_db=3.0,
                time_to_trigger_s=0.0,
                handover_execution_s=0.0,
            ),
        )
        self.assertEqual(result["summary"]["handover_completion_count"], 0)
        self.assertTrue(np.all(result["serving_ap_id"] == "AP-A"))

    def test_short_advantage_does_not_satisfy_ttt(self):
        result = run_two_ap(
            [-50.0] * 9,
            [-60.0, -60.0, -45.0, -45.0, -60.0, -45.0, -45.0, -60.0, -60.0],
            config=HandoverConfig(
                decision_filter="none",
                hysteresis_db=1.0,
                time_to_trigger_s=0.3,
                handover_execution_s=0.0,
            ),
        )
        self.assertEqual(result["summary"]["handover_trigger_count"], 0)
        self.assertEqual(result["summary"]["handover_completion_count"], 0)

    def test_sustained_advantage_triggers_after_ttt_and_execution_delay(self):
        result = run_two_ap(
            [-50.0] * 10,
            [-60.0, -60.0] + [-45.0] * 8,
            config=HandoverConfig(
                decision_filter="none",
                hysteresis_db=1.0,
                time_to_trigger_s=0.3,
                handover_execution_s=0.2,
            ),
        )
        trigger = next(
            event
            for event in result["events"]
            if event["event_type"] == "handover_trigger"
        )
        completion = next(
            event
            for event in result["events"]
            if event["event_type"] == "handover_complete"
        )
        self.assertAlmostEqual(trigger["condition_start_time_s"], 0.2)
        self.assertAlmostEqual(trigger["trigger_time_s"], 0.5)
        self.assertAlmostEqual(completion["completion_time_s"], 0.7)
        self.assertEqual(completion["from_ap_id"], "AP-A")
        self.assertEqual(completion["to_ap_id"], "AP-B")
        self.assertEqual(result["serving_ap_id"][5], "AP-A")
        self.assertEqual(result["serving_ap_id"][7], "AP-B")

    def test_emergency_branch_is_separate_from_normal_ttt(self):
        available = np.ones((2, 6), dtype=bool)
        available[0, 3:] = False
        result = run_two_ap(
            [-45.0] * 6,
            [-60.0] * 6,
            available=available,
            config=HandoverConfig(
                decision_filter="none",
                hysteresis_db=20.0,
                time_to_trigger_s=5.0,
                handover_execution_s=0.0,
                emergency_on_serving_unavailable=True,
            ),
        )
        trigger = next(
            event
            for event in result["events"]
            if event["event_type"] == "handover_trigger"
        )
        self.assertEqual(trigger["reason"], "serving_unavailable_emergency")
        self.assertEqual(result["serving_ap_id"][3], "AP-B")

    def test_front_and_rear_run_independent_real_candidate_state_machines(self):
        result = generate_dual_obm_handover(
            config=HandoverConfig(
                decision_filter="ewma",
                filter_tau_s=0.2,
                hysteresis_db=3.0,
                time_to_trigger_s=0.3,
                handover_execution_s=0.1,
            ),
            x_start=-600.0,
            x_end=600.0,
            v=20.0,
            train_length_m=200.0,
            simulation_step_s=0.01,
            report_interval_s=0.05,
            position_alignment_sigma_m=0.0,
            pointing_jitter_sigma_deg=0.0,
            trip_power_sigma_dB=0.0,
            receiver_noise_sigma_dB=0.0,
            sigma_shadow=0.0,
            K_linear=1.0e9,
            rssi_quantization_dB=0.0,
            seed=123,
        )
        for role in ("front_obm", "rear_obm"):
            self.assertEqual(
                result[role]["summary"]["initial_association_count"], 1
            )
            self.assertEqual(
                result[role]["summary"]["handover_completion_count"], 2
            )
            self.assertEqual(result[role]["serving_ap_id"][0], "AP-001")
            self.assertEqual(result[role]["serving_ap_id"][-1], "AP-003")
        front_completion_times = [
            event["completion_time_s"]
            for event in result["front_obm"]["events"]
            if event["event_type"] == "handover_complete"
        ]
        rear_completion_times = [
            event["completion_time_s"]
            for event in result["rear_obm"]["events"]
            if event["event_type"] == "handover_complete"
        ]
        self.assertFalse(
            np.array_equal(front_completion_times, rear_completion_times)
        )
        self.assertNotIn("combined_obm_rssi_dBm", result)
        self.assertNotIn("selected_obm", result)

    def test_configuration_rejects_nonphysical_values(self):
        with self.assertRaises(ValueError):
            normalise_handover_config({"filter_tau_s": 0.0})
        with self.assertRaises(ValueError):
            normalise_handover_config({"moving_average_window_s": 0.0})
        with self.assertRaises(ValueError):
            normalise_handover_config({"hysteresis_db": -1.0})
        with self.assertRaises(ValueError):
            normalise_handover_config({"time_to_trigger_s": -0.1})
        with self.assertRaises(ValueError):
            normalise_handover_config({"decision_filter": "mystery"})


if __name__ == "__main__":
    unittest.main()
