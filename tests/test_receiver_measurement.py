import unittest

import numpy as np

from receiver_measurement import (
    apply_receiver_reporting,
    causal_trailing_linear_average,
    dbm_to_mw,
    integrate_rssi_dbm_causally,
    mw_to_dbm,
)
from signal_generation import (
    TIME_DOMAIN_SAMPLING,
    generate_fault_rssi_pair,
    generate_rssi_simulation,
)


class ReceiverMeasurementTests(unittest.TestCase):
    def test_linear_power_average_is_not_dbm_average(self):
        result = integrate_rssi_dbm_causally(
            np.asarray([0.0, 20.0]),
            np.asarray([0.0, 1.0]),
            1.0,
        )
        self.assertAlmostEqual(float(result.averaged_power_mW[1]), 50.5)
        self.assertAlmostEqual(float(mw_to_dbm(result.averaged_power_mW)[1]), 17.03291378)
        self.assertNotAlmostEqual(float(mw_to_dbm(result.averaged_power_mW)[1]), 10.0)

    def test_causal_window_does_not_use_future_samples(self):
        result = causal_trailing_linear_average(
            np.asarray([1.0, 1.0, 100.0]),
            np.asarray([0.0, 1.0, 2.0]),
            2.0,
        )
        np.testing.assert_allclose(result.averaged_power_mW[:2], [1.0, 1.0])
        self.assertAlmostEqual(float(result.averaged_power_mW[2]), 25.75)

    def test_constant_signal_is_unchanged_on_irregular_time_grid(self):
        instantaneous = np.full(5, -63.25)
        result = integrate_rssi_dbm_causally(
            instantaneous,
            np.asarray([0.0, 0.01, 0.04, 0.11, 0.20]),
            0.05,
        )
        np.testing.assert_allclose(mw_to_dbm(result.averaged_power_mW), instantaneous)

    def test_reporting_quantizes_then_records_limits_and_raw_status(self):
        result = apply_receiver_reporting(
            np.asarray([-100.26, -19.74, -50.24]),
            receiver_sensitivity_dBm=-100.0,
            receiver_saturation_dBm=-20.0,
            rssi_quantization_dB=0.5,
        )
        np.testing.assert_allclose(result.quantized_rssi_dBm, [-100.5, -19.5, -50.0])
        np.testing.assert_allclose(result.reported_rssi_dBm, [-100.0, -20.0, -50.0])
        self.assertTrue(result.below_sensitivity_mask[0])
        self.assertTrue(result.saturation_mask[1])
        self.assertIn("raw_preserved", str(result.status[0]))
        self.assertIn("raw_preserved", str(result.status[1]))

    def test_missing_policy_preserves_raw_value_outside_report_array(self):
        raw = np.asarray([-110.0, -80.0])
        result = apply_receiver_reporting(
            raw,
            receiver_sensitivity_dBm=-100.0,
            receiver_saturation_dBm=-20.0,
            rssi_quantization_dB=0.0,
            below_sensitivity_policy="missing_report_preserve_raw_status",
        )
        self.assertTrue(np.isnan(result.reported_rssi_dBm[0]))
        self.assertAlmostEqual(float(raw[0]), -110.0)
        self.assertTrue(result.missing_mask[0])

    def test_time_domain_metadata_exposes_causal_receiver_support(self):
        result = generate_rssi_simulation(
            x_start=-10.0,
            x_end=10.0,
            v=20.0,
            sampling_mode=TIME_DOMAIN_SAMPLING,
            simulation_step_s=0.01,
            report_interval_s=0.05,
            measurement_window_s=0.05,
            receiver_noise_sigma_dB=0.0,
            rssi_quantization_dB=0.0,
            receiver_sensitivity_dBm=None,
            receiver_saturation_dBm=None,
            seed=21,
            return_metadata=True,
        )
        metadata = result[6]
        self.assertEqual(metadata["receiver_integration_coordinate_unit"], "s")
        self.assertEqual(
            metadata["receiver_integration_startup_policy"],
            "partial_available_history",
        )
        self.assertAlmostEqual(
            float(metadata["receiver_integration_support_width_median"]),
            0.05,
        )
        self.assertEqual(
            len(metadata["receiver_report_status"]),
            len(result[0]),
        )

    def test_global_attenuation_is_preserved_by_linear_integration(self):
        _, healthy, faulty = generate_fault_rssi_pair(
            "全链路功率衰减",
            {"atten_dB": 8.0},
            x_start=-20.0,
            x_end=20.0,
            v=20.0,
            sampling_mode=TIME_DOMAIN_SAMPLING,
            simulation_step_s=0.01,
            report_interval_s=0.05,
            measurement_window_s=0.05,
            receiver_noise_sigma_dB=0.0,
            rssi_quantization_dB=0.0,
            receiver_sensitivity_dBm=None,
            receiver_saturation_dBm=None,
            seed=9,
        )
        np.testing.assert_allclose(faulty - healthy, -8.0, atol=1e-10)

    def test_dbm_mw_roundtrip(self):
        values = np.asarray([-100.0, -50.0, 0.0, 20.0])
        np.testing.assert_allclose(mw_to_dbm(dbm_to_mw(values)), values)


if __name__ == "__main__":
    unittest.main()
