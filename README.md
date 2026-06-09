# AppliedML2026

This repository contains our Applied ML 2026 project: **country prediction from GeoGuessr street-view images** using convolutional neural networks.

## Project at a glance
- **Task:** classify images from **85 countries**
- **Dataset:** ~20,062 train images and ~5,084 test images (GeoGuessr-countries)
- **Main models:** pretrained ResNet variants (especially ResNet34/ResNet50)
- **Pipeline:** image preprocessing + on-the-fly augmentation + fine-tuning + evaluation with Top-1/Top-5 and per-class metrics

## What we found
- On-the-fly augmentation was important; precomputed augmentation caused more overfitting.
- European countries were generally harder to separate than visually distinct countries.
- Class weighting and additional tuning changed behavior, but results were mixed.
- A cluster-based setup (countries grouped by climate/geography features) reduced task complexity and improved separability for some groups.

### Key reported results
- **ResNet50 country-level model:** Top-1 **0.487**, Top-5 **0.779**  
  (see `country_cnn/outputs/models/resnet50_test_summary.csv`)
- **Black-box car-artifact experiment (presentation):** Top-1 **0.502**, Top-5 **0.813**

## Repository structure
- `country_cnn/src/` – model creation, data loading, training, evaluation
- `country_cnn/outputs/models/` – saved models, summaries, classification reports, confusion matrices
- `country_cnn/outputs/clustering_results/` – cluster analyses and plots
- `ML rigtig.pptx.pdf` – presentation with methods, experiments, and conclusions

## Notes
- Large datasets are not stored in this repository (see project links in the presentation for data access).