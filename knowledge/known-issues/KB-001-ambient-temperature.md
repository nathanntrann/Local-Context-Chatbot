---
title: "KB-001: Lighting and Ambient Temperature Effects"
category: known-issues
tags: [lighting, temperature, ambient, drift, environmental, known-issue]
issue_id: KB-001
severity: medium
---

# KB-001: Lighting and Ambient Temperature Effects

## Issue

Thermal inspection results drift throughout the day as factory ambient temperature changes. Morning shift sees lower rejection rates than afternoon shift.

## Root Cause

Thermal cameras measure relative temperature differences. As ambient temperature rises:
- The absolute temperature of both the seal and surrounding material increases
- The *contrast* between sealed and unsealed areas may decrease
- Fixed thresholds that worked at 18°C ambient may be too sensitive at 28°C

## Impact

- 2-5% increase in false positive rate during warm periods
- Operators may over-adjust thresholds to compensate, causing missed defects later

## Workarounds

1. **Morning calibration**: Recalibrate thermal reference at the start of each shift
2. **Wider tolerance**: Adjust threshold by +0.03 to +0.05 for afternoon running
3. **Monitor ambient**: Track ambient temperature alongside rejection rate to correlate

## Permanent Fix

Implement adaptive thresholds that adjust based on ambient temperature sensor input. Requires system-level integration with a temperature probe.

## Affected Versions

All versions using fixed thermal thresholds.
