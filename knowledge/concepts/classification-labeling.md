---
title: Image Classification and Labeling
category: concepts
tags: [classification, labeling, PASS, FAULT, dataset, machine-learning, training]
---

# Image Classification and Labeling

## Label Definitions

- **PASS**: Image shows a thermal seal that meets quality requirements. Uniform heat distribution, consistent seal width, no visible defects.
- **FAULT**: Image shows a thermal seal with one or more defects — cold spots, partial sealing, contamination, burn-through, or significant irregularity.

## Labeling Best Practices

1. **Be consistent** — establish clear criteria before labeling and follow them for every image
2. **When in doubt, mark for review** — ambiguous images should be reviewed by a second person
3. **Include edge cases** — borderline images help the model learn the decision boundary
4. **Balance the dataset** — aim for roughly equal PASS and FAULT counts for training
5. **Track labeling decisions** — document why borderline images were labeled as they were

## Common Labeling Mistakes

- **Ignoring subtle defects** → Labeling a clearly defective seal as PASS because the defect is small
- **Being too strict** → Labeling normal process variation as FAULT
- **Inconsistency between labelers** → Different people apply different standards
- **Not considering the full image** → Focusing on one region and missing defects elsewhere

## Dataset Quality Checks

- Review class balance (PASS vs FAULT ratio)
- Randomly sample from each class and verify labels
- Look for images that are ambiguous or near the decision boundary
- Check for duplicate or near-duplicate images
- Verify image quality (focus, exposure, noise levels)
