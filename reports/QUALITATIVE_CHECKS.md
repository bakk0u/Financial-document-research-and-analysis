# Qualitative checks outside the benchmark

These are manual sanity probes, not extra benchmark examples. They are not included in reported test scores, and no model tuning followed them.

## Retained failure

Input: “Operating profit increased as revenue grew and costs declined.”

The deployed calibrated model predicted **negative (83.0%)**, neutral (0.2%), positive (16.8%). A natural financial interpretation of rising profit/revenue and falling costs is positive; the model's negative prediction is a failure. This illustrates brittleness around mixed directional words and why aggregate held-out performance does not guarantee correct individual outputs. The example remains accessible in the interface rather than being silently removed.

Do not infer the precise internal cause from this one observation. Establishing whether cost-direction language causes the failure would require a controlled contrast set.

## Interface checks

- Loaded the actual saved adapter checkpoint for browser inference.
- Displayed class probabilities, model identity, runtime and audit event reference.
- Displayed all four held-out baseline/model rows from measured evaluation.
- Retrieved an Apple filing passage and transferred it directly into the sentiment analysis workspace.

Full per-example benchmark errors can be inspected in `sentiment-predictions.json` by selecting rows where `truth != prediction`.
