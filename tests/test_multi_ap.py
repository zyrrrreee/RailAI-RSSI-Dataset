import unittest

import numpy as np

from multi_ap import (
    DEFAULT_WAYSIDE_APS,
    WaysideAP,
    generate_multi_ap_dual_obm_candidates,
)


COMMON = {
    "x_start": -600.0,
    "x_end": 600.0,
    "v": 20.0,
    "simulation_step_s": 0.01,
    "report_interval_s": 0.05,
    "position_alignment_sigma_m": 0.75,
    "pointing_jitter_sigma_deg": 0.0,
    "trip_power_sigma_dB": 0.0,
    "receiver_noise_sigma_dB": 0.0,
    "sigma_shadow": 0.0,
    "K_linear": 1.0e9,
    "rssi_quantization_dB": 0.0,
}


class MultiApCandidateTests(unittest.TestCase):
    def test_candidate_matrix_keeps_every_ap_for_both_obms(self):
        result = generate_multi_ap_dual_obm_candidates(
            aps=DEFAULT_WAYSIDE_APS, train_length_m=200.0, seed=123, **COMMON
        )
        sample_count = len(result["time_s"])
        self.assertEqual(
            result["front_obm"][
                "candidate_reported_rssi_matrix_dBm"
            ].shape,
            (3, sample_count),
        )
        self.assertEqual(
            result["rear_obm"][
                "candidate_reported_rssi_matrix_dBm"
            ].shape,
            (3, sample_count),
        )
        self.assertEqual(result["ap_ids"], ("AP-001", "AP-002", "AP-003"))

    def test_all_ap_links_share_one_train_clock_and_geometry(self):
        result = generate_multi_ap_dual_obm_candidates(seed=123, **COMMON)
        for ap_id in result["ap_ids"]:
            link = result["ap_links"][ap_id]
            np.testing.assert_array_equal(link["time_s"], result["time_s"])
            np.testing.assert_array_equal(
                link["train_center_position_m"],
                result["train_center_position_m"],
            )
            self.assertAlmostEqual(
                link["metadata"]["common_position_alignment_offset_m"],
                result["metadata"]["global_position_alignment_offset_m"],
            )

    def test_each_identical_ap_has_ideal_peak_near_its_track_position(self):
        result = generate_multi_ap_dual_obm_candidates(seed=123, **COMMON)
        front_x = result["front_obm"]["x_m"]
        for row, ap in enumerate(DEFAULT_WAYSIDE_APS):
            ideal = result["front_obm"][
                "candidate_ideal_rssi_matrix_dBm"
            ][row]
            peak_x = float(front_x[int(np.argmax(ideal))])
            self.assertLessEqual(abs(peak_x - ap.track_position_m), 15.0)

    def test_ideal_strongest_regions_follow_ap_order(self):
        result = generate_multi_ap_dual_obm_candidates(seed=123, **COMMON)
        identifiers = result["front_obm"][
            "reference_ideal_strongest_ap_id"
        ]
        compressed = [
            str(value)
            for index, value in enumerate(identifiers)
            if index == 0 or value != identifiers[index - 1]
        ]
        self.assertEqual(compressed, ["AP-001", "AP-002", "AP-003"])

    def test_reference_argmax_is_not_exposed_as_serving_or_handover(self):
        result = generate_multi_ap_dual_obm_candidates(seed=123, **COMMON)
        self.assertEqual(
            result["metadata"]["serving_ap_policy"], "not_implemented"
        )
        self.assertNotIn("serving_ap_id", result["front_obm"])
        self.assertNotIn("handover_events", result["front_obm"])
        self.assertIn(
            "reference_report_strongest_ap_id", result["front_obm"]
        )

    def test_ap_candidate_retains_dual_antenna_branch_evidence(self):
        result = generate_multi_ap_dual_obm_candidates(seed=123, **COMMON)
        candidate = result["front_obm"]["candidates_by_ap"]["AP-002"]
        self.assertIn("antenna_1_candidate_raw_rssi_dBm", candidate)
        self.assertIn("antenna_2_candidate_raw_rssi_dBm", candidate)
        self.assertIn("selected_antenna", candidate)

    def test_fixed_seed_reproduces_all_candidate_reports(self):
        first = generate_multi_ap_dual_obm_candidates(seed=456, **COMMON)
        second = generate_multi_ap_dual_obm_candidates(seed=456, **COMMON)
        for role in ("front_obm", "rear_obm"):
            np.testing.assert_array_equal(
                first[role]["candidate_reported_rssi_matrix_dBm"],
                second[role]["candidate_reported_rssi_matrix_dBm"],
            )

    def test_ap_configuration_validation_and_ambiguity(self):
        with self.assertRaises(ValueError):
            generate_multi_ap_dual_obm_candidates(
                aps=[WaysideAP("AP-1", 0.0)], seed=123, **COMMON
            )
        with self.assertRaises(ValueError):
            generate_multi_ap_dual_obm_candidates(
                aps=[WaysideAP("same", 0.0), WaysideAP("same", 300.0)],
                seed=123,
                **COMMON,
            )
        with self.assertRaises(ValueError):
            generate_multi_ap_dual_obm_candidates(
                seed=123, antenna_x=10.0, **COMMON
            )

    def test_front_and_rear_remain_separate_for_every_ap(self):
        result = generate_multi_ap_dual_obm_candidates(seed=123, **COMMON)
        np.testing.assert_allclose(
            result["front_obm"]["x_m"] - result["rear_obm"]["x_m"],
            200.0,
        )
        self.assertNotIn("selected_obm", result)
        self.assertNotIn("combined_obm_rssi_dBm", result)


if __name__ == "__main__":
    unittest.main()
