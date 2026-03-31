---
title: Troubleshooting Inconsistent Results
category: troubleshooting
tags: [inconsistent, variation, unstable, results, repeatability]
---

# Inconsistent Inspection Results

When the same product type gets different results on different runs, or results vary significantly within a single batch.

## Symptoms

- Same product passes on one run, fails on the next
- Rejection rate fluctuates without process changes
- Results vary by time of day or shift

## Diagnosis Steps

1. **Isolate the variable** — is variability in the product, the system, or the environment?
   - Run the same known-good sample 10 times — if results vary, it's the system
   - If results are consistent with the same sample, it's product variation

2. **Check timing** — is the camera triggering at the same point in the process?
   - Inconsistent trigger timing captures different thermal states

3. **Check lighting/environment** — thermal cameras are affected by:
   - Ambient temperature changes
   - Air currents from HVAC
   - Nearby heat sources (motors, lights)

4. **Check conveyor speed** — speed variations change the thermal state at capture time

5. **Review the reference/model** — was it trained on a representative sample?

## Common Fixes

- **Stabilize trigger timing** — use a consistent conveyor speed and trigger delay
- **Shield the inspection area** — reduce drafts and radiant heat from nearby equipment
- **Recalibrate** — update the thermal reference if ambient conditions have changed permanently
- **Improve region of interest (ROI)** — ensure the inspection area is consistently placed on the product
- **Add averaging** — inspect multiple frames and aggregate results to reduce noise
