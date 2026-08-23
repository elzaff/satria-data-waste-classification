# SATRIA DATA Waste Image Classification

Competition solution for recyclable-waste image classification. The selected system uses a SigLIP2 NaFlex vision-language backbone with staged fine-tuning and balanced logistic regression.

## Contents

- `final514/`: selected Final514 experiment, recipe, Modal pipeline, and result summary.
- `clean-pipeline/`: cleaned reusable training/inference pipeline and manifest.

## Results Context

The documented Final514 validation configuration reports 42 validation errors and Macro-F1 0.993515. These numbers come from the project's experiment records and use the validation protocol described in the included reports. They are not claimed as independent test scores here.

## Requirements

The selected pipeline was designed for GPU execution through [Modal](https://modal.com). It expects official competition images, model caches, and secrets to be mounted through the execution environment. Raw datasets, embeddings, checkpoints, and submission files are intentionally excluded.

See `final514/README.md`, `final514/recipe.json`, and `clean-pipeline/README.md` for implementation details.
