---
title: Getting Started with Inspection Setup
category: procedures
tags: [setup, calibration, getting-started, procedure, recipe]
---

# Getting Started with Inspection Setup

## Pre-Setup Checklist

- [ ] Camera is mounted and focused on the seal area
- [ ] Lighting/IR source is positioned correctly
- [ ] Conveyor speed is set and stable
- [ ] Sample products available (known good + known bad)
- [ ] Software is connected to the camera

## Setup Procedure

### Step 1: Capture Reference Images
- Run 20-30 known good products
- Save thermal images as PASS baseline
- Note ambient temperature and time

### Step 2: Capture Defect Images
- Create or collect products with known defects
- Run them through the system
- Save thermal images as FAULT examples
- Include different defect types if possible

### Step 3: Define Region of Interest (ROI)
- Set the inspection area to cover the full seal
- Include a small margin beyond the seal edges
- Avoid including non-relevant areas (reduces noise)

### Step 4: Set Initial Thresholds
- Start with moderate sensitivity (0.5)
- Run a mixed batch of good and bad samples
- Check detection rate and false positive rate
- Adjust in small increments

### Step 5: Validate
- Run a full production batch
- Review all rejections manually
- Calculate: true positives, false positives, false negatives
- Target: >95% detection rate, <3% false positive rate

### Step 6: Document and Lock
- Record all parameters with the date
- Note ambient conditions during validation
- Lock configuration to prevent accidental changes
