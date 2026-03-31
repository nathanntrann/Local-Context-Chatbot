---
title: Troubleshooting False Positives
category: troubleshooting
tags: [false-positive, rejection, troubleshooting, threshold, calibration]
---

# False Positives — Good Parts Being Rejected

False positives (good parts incorrectly classified as FAULT) are one of the most common and disruptive issues in thermal inspection.

## Quick Diagnosis Checklist

1. **Check the rejection rate trend** — is it sudden or gradual?
   - Sudden spike → likely environmental or mechanical change
   - Gradual increase → possible sensor drift or material change

2. **Review rejected images** — do they look like good seals to a human?
   - If yes → threshold is too sensitive, or lighting/temperature changed
   - If no → they may be real defects — investigate the sealing process

3. **Check ambient temperature** — has it changed significantly?
   - Thermal inspection is relative — a 10°C ambient shift changes absolute readings

4. **Check camera/sensor** — is the lens clean? Any condensation?

5. **Check product alignment** — are parts positioned consistently?
   - Misalignment can cause the inspection region to miss the seal area

## Common Root Causes

| Cause | How to Verify | Fix |
|-------|--------------|-----|
| Threshold too low | Review confidence scores — bunched near threshold | Raise threshold by 0.05, re-evaluate |
| Ambient temp change | Compare morning vs afternoon rejection rates | Recalibrate or use adaptive thresholds |
| Sensor contamination | Inspect camera lens | Clean lens, check for condensation |
| Product variation | Compare rejected parts across batches | Update reference model or widen tolerance |
| Material change | Check incoming material lot numbers | Retrain or adjust for new material |

## When to Escalate

- False positive rate exceeds 5% sustained
- Threshold adjustments don't reduce the rate
- Rejected images show no visible defect pattern
- Issue coincides with a software or hardware change
