---
title: Camera and Optics Best Practices
category: concepts
tags: [camera, optics, IR, hardware, lens, focus, NETD]
---

# Camera and Optics Best Practices

The infrared camera is the foundation of thermal seal inspection. Proper setup and maintenance are essential for reliable results.

## Key Camera Parameters

### NETD (Noise Equivalent Temperature Difference)
- Measures the camera's thermal sensitivity (ability to detect small temperature differences)
- Lower is better — typical industrial cameras: 30–80 mK
- A camera with high NETD (> 100 mK) may miss subtle defects

### Frame Rate
- Must match or exceed conveyor speed to avoid motion blur
- Typical: 30–60 fps for moderate-speed lines
- High-speed lines (> 100 packs/min) may need 120+ fps or triggered capture

### Resolution
- Higher resolution allows more detail per pixel on the seal area
- Common: 320×240 (basic), 640×480 (standard), 1024×768 (high-end)
- Match resolution to the smallest defect you need to detect

## Lens Selection

- **Field of view (FOV)** must cover the full seal width with margin
- Too narrow: seals get clipped at the edges
- Too wide: fewer pixels per mm of seal, less detail
- Use a lens that places the seal area across at least 50% of the sensor width

## Focus

- Auto-focus is generally unreliable for thermal — use manual focus
- Focus on the seal plane, not the product surface
- Re-check focus after any camera or mounting change
- Use a thermal target (e.g. warm resistor or heated metal strip) to confirm sharpness

## Mounting

- Mount camera perpendicular to the seal plane for consistent geometry
- Avoid vibration — use rigid brackets, not flexible arms
- Working distance: 200–500 mm typical; shorter = more detail but tighter alignment

## Maintenance

- Clean lens with appropriate IR-safe lens cleaner (not regular glass cleaner)
- Check for condensation in cold/humid environments
- Verify NUC (non-uniformity correction) is running — most cameras do this automatically
- Replace camera if NETD degrades noticeably over time
