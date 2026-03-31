---
title: Understanding Inspection Thresholds
category: parameters
tags: [threshold, sensitivity, parameters, tuning, configuration]
---

# Inspection Thresholds

Thresholds define the boundary between PASS and FAULT classifications. Setting thresholds correctly is critical for balancing detection sensitivity against false positive rates.

## Key Threshold Parameters

### Detection Threshold
- **What it does**: Sets the minimum confidence score required to flag a defect
- **Range**: 0.0 to 1.0 (typical: 0.4 to 0.7)
- **Effect of raising**: Fewer detections — reduces false positives but may miss real defects
- **Effect of lowering**: More detections — catches more defects but increases false positives

### Sensitivity
- **What it does**: Controls how responsive the system is to small thermal variations
- **High sensitivity**: Detects subtle defects but sensitive to normal process variation
- **Low sensitivity**: Only catches obvious defects, more tolerant of noise

## Common Mistakes

1. **Setting threshold too low** after seeing a missed defect — causes flood of false positives
2. **Not accounting for ambient temperature changes** — thresholds that work at 20°C may fail at 30°C
3. **Copy-pasting thresholds between products** — different materials have different thermal signatures
4. **Adjusting sensitivity instead of investigating root cause** — masking a real process issue

## Tuning Guidance

1. Start with a moderate threshold (0.5) and run a batch
2. Review false positives and false negatives
3. Adjust in small increments (0.05) and re-evaluate
4. Always validate with known good and known bad samples
5. Document the threshold and the reason for the change
