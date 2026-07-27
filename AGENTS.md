# RailAI-RSSI-Dataset collaboration rules

1. This repository publishes reproducible simulated railway RSSI data and the
   minimum code needed to generate, validate, convert, and benchmark it.
2. Keep the canonical hierarchy `Scenario -> Run -> Sample -> Observations`.
   `Observations` are rows inside a sample, not a fourth identifier level.
3. Never describe simulated data as measurements from a real railway line.
   Document every parameter as literature-derived, equipment-derived, calibrated,
   or an explicit engineering assumption.
4. Preserve healthy/fault paired runs by reusing the same frozen random world.
   Fault mechanisms must change only the intended physical component.
5. Never split neighboring rows or paired runs across train/validation/test sets.
   Official splits are grouped by run, pair, or scenario to prevent leakage.
6. Do not commit generated full datasets, trained models, caches, credentials, or
   local logs. Publish large versioned artifacts through Releases with SHA-256
   checksums.
7. Before changing physical models or dataset fields, explain the current
   behavior, problem, proposed change, evidence, unit, parameter range, and
   expected sensitivity.
8. Maintain backward compatibility where practical and update the dataset card,
   field dictionary, tests, and changelog whenever the public contract changes.
