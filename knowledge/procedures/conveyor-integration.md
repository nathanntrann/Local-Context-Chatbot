---
title: Conveyor Integration and Timing
category: procedures
tags: [conveyor, integration, timing, trigger, synchronization, speed]
---

# Conveyor Integration and Timing

Proper integration between the conveyor system and the thermal inspection camera is critical for consistent, reliable results.

## Trigger Modes

### Continuous Capture
- Camera runs at a fixed frame rate
- Simple to set up, but may miss products if timing drifts
- Works for slow lines (< 20 packs/min)

### External Trigger (Recommended)
- A sensor (photoelectric, proximity, or encoder pulse) triggers each capture
- Ensures one image per product at the correct position
- Requires wiring the trigger signal to the camera or frame grabber

### Encoder-Based
- Encoder on the conveyor drives capture at precise intervals
- Best for variable-speed lines
- Requires encoder input on the camera or capture card

## Timing Considerations

### Capture Delay
- Time between seal station and camera must be consistent
- If conveyor speed changes, adjust the trigger delay to compensate
- The seal should be fully formed but still warm when captured — typically 100–500 ms after sealing

### Exposure Time
- Too long: motion blur on fast lines
- Too short: noisy images with poor thermal contrast
- Start with 1–5 ms and adjust based on image quality

## Common Integration Problems

| Problem | Symptom | Fix |
|---------|---------|-----|
| Speed mismatch | Images offset or partial seals visible | Re-calibrate trigger delay to match current speed |
| Missing triggers | Gaps in inspection — some products not captured | Check sensor alignment and sensitivity |
| Double triggers | Same product inspected twice | Add debounce delay or adjust sensor position |
| Vibration | Blurry images at high speed | Improve mounting rigidity, reduce exposure time |
| Inconsistent position | Seal not centered in frame | Move sensor closer to camera to reduce drift |

## Speed Change Protocol

When conveyor speed changes:
1. Verify trigger timing is still correct (seal centered in frame)
2. Check exposure time is still adequate (no blur)
3. Re-validate thresholds — faster speeds may cool seals faster
4. Run a batch of known-good and known-bad samples to confirm performance
