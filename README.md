# FLAME AI 2024 Paper Reproducibility Repository

This repository contains the code and archived model outputs needed to reproduce the main qualitative and quantitative figures from the **FLAME AI 2024** paper.

The emphasis here is on **figure regeneration**, not on providing a single end-to-end training pipeline for every model. Several models compared in the paper were originally trained and/or inferred in external Kaggle notebooks or separate GitHub repositories. This repository collects the prediction outputs needed for the paper figures and provides lightweight scripts for plotting and evaluation.

## Repository purpose

This repository is intended to:

- reproduce the main paper figures from saved model outputs
- provide a transparent mapping between compared models and their outputs
- keep the paper artifact lightweight and easy to navigate
- document where each model came from and how it can be retrained or re-run elsewhere

## Dataset overview

The **FLAME AI 2024** dataset is a spatiotemporal fire prediction benchmark used to compare multiple forecasting approaches on a shared set of cases. In this repository, we include the processed artifacts needed to regenerate selected figures from the paper, rather than the full raw training pipeline for every model.

## Download model outputs
The `model_outputs/` directory needs to be downloaded before the other plotting scripts can work. To download and extract it:

```bash
python3 download_model_outputs.py
```

## Model provenance and Source links

This repository archives the prediction outputs and lightweight plotting/evaluation scripts used to reproduce the figures in the paper.

These links point to the original competition writeups, Kaggle notebooks, or external repositories associated with each model.

### Competition models

- **Mixed**  
  Source: [Ajay Asaithambi — LB first place solution](https://www.kaggle.com/competitions/2024-flame-ai-challenge/writeups/ajay-asaithambi-lb-first-place-solution)

- **Latent Loop**  
  Source: [Jobayer Hossain — FLAME AI second place solution](https://www.kaggle.com/competitions/2024-flame-ai-challenge/writeups/jobayer-hossain-flame-ai-second-place-solution)

- **SwinUNet**  
  Source: [Rafał Pawłowski — segmentation approach](https://www.kaggle.com/competitions/2024-flame-ai-challenge/writeups/rafa-paw-owski-segmentation-approach)

- **MultiResUNet**  
  Source: [Zhuoqun Li — FLAME_AI GitHub repository](https://github.com/lizhuoq/FLAME_AI)

- **Conv-TT-LSTM**  
  Source: [Thomas Dubail — Kaggle notebook](https://www.kaggle.com/code/thomasdubail/2024-flame-third-try-submission/)

### Baseline model links

- **Baseline**  
  Source: [FLAME AI 2024 baseline model](https://www.kaggle.com/code/jenzenho/flame-ai-2024-baseline-model)

- **Baseline + Otsu**  
  Source: [FLAME AI 2024 baseline with Otsu](https://www.kaggle.com/code/jenzenho/flame-ai-2024-baseline-with-otsu)

- **Baseline 4-fold**  
  Source: [FLAME AI 2024 baseline 4-fold](https://www.kaggle.com/code/jenzenho/flame-ai-2024-baseline-4fold)

- **Baseline 4-fold + Otsu (4x smaller)**  
  Source: [FLAME AI 2024 baseline 4-fold 4x smaller Otsu](https://www.kaggle.com/code/jenzenho/flame-ai-2024-baseline-4fold-4xsmaller-otsu)

- **Baseline 4-fold + Otsu (25 epochs)**  
  Source: [Baseline 4-fold 25 epochs ensemble rollout Otsu](https://www.kaggle.com/code/jenzenho/baseline-4fold-25epochs-ensemblerollout-otsu)
