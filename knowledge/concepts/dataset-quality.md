---
title: Understanding Dataset Quality and Labeling Metrics
category: concepts
tags: [dataset, quality, metrics, class-balance, mislabel, accuracy, statistics]
---

# Dataset Quality and Labeling Metrics

A high-quality labeled dataset is essential for building trust in the inspection system. This article explains key metrics used to evaluate dataset health.

## Class Balance

| Metric | Healthy Range | Warning |
|--------|--------------|---------|
| Pass ratio | 60–90% | < 50% may indicate over-labeling faults; > 95% may mean faults are being missed |
| Fault ratio | 10–40% | < 5% may be too few examples to learn from; > 50% is unusual |

Class imbalance is normal in inspection — most seals should be good. But extreme imbalance reduces the system's ability to learn fault patterns.

## Label Accuracy

**Mislabel rate**: percentage of images whose label disagrees with expert review.

- < 2%: Excellent — normal for careful human labeling
- 2–5%: Acceptable — common in high-volume labeling
- 5–10%: Concerning — investigate labeling process
- > 10%: Dataset unreliable — re-label with stricter QA

Use `find_suspicious_labels` to estimate mislabel rates via AI review.

## Image Quality Metrics

- **Resolution consistency**: all images should have the same dimensions (different sizes may indicate camera or crop changes)
- **File size variation**: large variance may indicate different compression, exposure, or capture modes
- **Blank or corrupt images**: zero-byte or very small files should be flagged and removed

Use `get_dataset_statistics` to check these metrics.

## What Makes a Good Training / Evaluation Dataset

1. **Representative**: covers the full range of good seals AND real defect types
2. **Balanced enough**: at least 20–30 fault examples for meaningful evaluation
3. **Correctly labeled**: < 5% mislabel rate
4. **Consistent capture**: same camera settings, similar timing, stable lighting
5. **Documented**: labeling criteria, who labeled it, when, and any known issues
