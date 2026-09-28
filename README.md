# Uncertainty-Aware Deep Learning

A reproducible PyTorch project exploring **predictive confidence, Monte Carlo Dropout and calibration** for neural-network classification.

A model can be accurate and still be dangerously overconfident. This project looks beyond accuracy and asks a second question: **when should the model be uncertain?**

## What this project demonstrates

- Neural-network training in PyTorch
- Deterministic vs Monte Carlo Dropout inference
- Predictive entropy and mutual-information uncertainty
- Expected Calibration Error (ECE) and Brier score
- Reliability analysis
- Selective prediction: accuracy when uncertain predictions are rejected
- Reproducible experiments on a public dataset
- Unit-tested uncertainty and calibration utilities

## Experiment

The default experiment uses the public **scikit-learn Breast Cancer Wisconsin diagnostic dataset**. It is small enough to reproduce quickly while still allowing uncertainty and calibration behaviour to be studied.

The dataset is split into train, validation and test sets. Features are standardized using statistics learned from the training set only.

## Model

A multilayer perceptron uses dropout during training. At ordinary inference, dropout is disabled and one probability vector is produced.

For **MC Dropout**, dropout remains active at inference and the same sample is passed through the network multiple times:

```text
x -> stochastic forward pass 1 -> p1
  -> stochastic forward pass 2 -> p2
  -> ...
  -> stochastic forward pass T -> pT
                              |
                              v
                     predictive mean
                     predictive entropy
                     mutual information
```

Variation across stochastic predictions provides an approximate measure of epistemic uncertainty.

## Metrics

**Accuracy** measures classification correctness.

**Expected Calibration Error (ECE)** compares predicted confidence with empirical accuracy across confidence bins.

**Brier score** measures the squared error of predicted probabilities.

**Predictive entropy** captures total uncertainty in the mean predictive distribution.

**Mutual information** separates uncertainty caused by disagreement between stochastic model passes and is used here as an epistemic-uncertainty signal.

## Selective prediction

A useful uncertainty estimate should help identify cases on which the model is less reliable. The experiment therefore ranks examples by uncertainty and reports accuracy as progressively more uncertain predictions are rejected.

This is more informative than claiming that uncertainty is useful simply because an uncertainty number can be computed.

## Run

```bash
git clone https://github.com/amrabah/uncertainty-aware-deep-learning.git
cd uncertainty-aware-deep-learning

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.experiment
pytest
```

The experiment writes metrics to `artifacts/metrics.json`.

## Repository structure

```text
src/
  data.py          # dataset loading, splitting and scaling
  model.py         # PyTorch MLP
  train.py         # training and deterministic prediction
  uncertainty.py   # MC Dropout and uncertainty decomposition
  metrics.py       # calibration and selective-prediction metrics
  experiment.py    # reproducible end-to-end experiment
tests/
```

## Methodological note

MC Dropout is an approximate Bayesian technique, not a guarantee that uncertainty is perfectly calibrated or that a model is safe under distribution shift. Uncertainty estimates should themselves be evaluated against the intended use case.

## Background

This independent portfolio project is inspired by my previous research experience with neural networks, robustness and Bayesian dropout. It uses only a public dataset and newly written code.

## Author

**Aya Mrabah** — AI Engineer / Data Scientist

Interests: applied machine learning, uncertainty estimation, explainable AI and robust AI systems.
