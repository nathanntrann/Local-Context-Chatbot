---
title: "KB-002: Sensor Contamination Causing False Faults"
category: known-issues
tags: [sensor, contamination, lens, cleaning, false-positive, known-issue]
---

# KB-002: Sensor Contamination Causing False Faults

## Symptoms
- Gradual increase in FAULT detections over days or weeks
- False faults appear in the same image region consistently
- Cleaning the lens resolves the issue temporarily

## Root Cause

Dust, condensation, or residue from the packaging process accumulates on the IR camera lens or protective window. This creates a localized temperature offset in the image, causing the system to interpret it as a seal defect.

## Diagnosis

1. Visually inspect the camera lens and any protective window for spots, smudges, or condensation
2. Look at recent false faults — are they clustered in the same image region? If yes, contamination is likely
3. Capture an image of a uniform surface (e.g. a wall) — non-uniform spots indicate lens issues
4. Check if the issue correlates with humidity, steam, or production debris

## Workaround

- Clean the lens with IR-safe lens cleaner and a microfiber cloth
- If using a protective window, clean or replace it
- Increase the cleaning schedule in dusty or humid environments

## Permanent Fix

- Install a protective window / enclosure with air purge (positive pressure) to keep contaminants off the lens
- Schedule cleaning as part of daily maintenance (start of shift)
- Add a reference surface check to the startup procedure — capture a flat target and compare to baseline

## Prevention

- Position camera to minimize exposure to packaging dust, steam, or overspray
- Use a lens hood to reduce direct contamination
- Monitor false-positive rates over time — a rising trend often points to contamination before it's visible
