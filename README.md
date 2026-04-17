# FLAME AI 2024 Paper Reproducibility Repository

This repository contains the code and archived model outputs needed to reproduce the main qualitative and quantitative figures from the **FLAME AI 2024** paper.

The emphasis here is on **figure regeneration**, not on providing a single end-to-end training pipeline for every model. Several models compared in the paper were originally trained and/or inferred in external Kaggle notebooks or separate GitHub repositories. This repository collects the prediction outputs needed for the paper figures and provides lightweight scripts for plotting and evaluation.

## Repository purpose

This repository is intended to:

- reproduce the main paper figures from saved model outputs
- provide a transparent mapping between compared models and their outputs
- keep the paper artifact lightweight and easy to navigate
- document where each model came from and how it can be retrained or re-run elsewhere

## Download model outputs
The `model_outputs/` directory needs to be downloaded before the other plotting scripts can work. To download and extract it:

```bash
python3 download_model_outputs.py

## Dataset overview

The **FLAME AI 2024** dataset is a spatiotemporal fire prediction benchmark used to compare multiple forecasting approaches on a shared set of cases. In this repository, we include the processed artifacts needed to regenerate selected figures from the paper, rather than the full raw training pipeline for every model.
