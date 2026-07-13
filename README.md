# Cross-Modal Knowledge Distillation for Acoustic Pedestrian Detection

Companion code for the DCASE 2026 submission of the same title, proposing **Trust-Filtered Distillation (TFD)**.

Audio-only pedestrian detection on [ASPED v.a](https://huggingface.co/datasets/urbanaudiosensing/ASPED) (urban pedestrian areas closed to vehicular traffic) via cross-modal knowledge distillation from a video teacher (Mask2Former) to an audio student (VGGish + Transformer). Under the dataset's severe class imbalance (91.5% no-pedestrian / 8.5% pedestrian-present), the KD gradient is dominated by the majority class and shifts the student's operating point against the rare pedestrian class. **Trust-Filtered Distillation (TFD)** gates the KD loss per sample, disabling it only on the minority samples the teacher misclassifies, and is the only configuration in our eleven-way LOSO comparison to reach statistical significance (paired *t*-test *p* = 0.033 / 0.026, Cohen's *d* = 1.60 / 1.72).

**TL;DR.** Although the video teacher is far more accurate than the audio student (89.9% vs. 72.9% LOSO macro accuracy), KD transfers little discriminative knowledge across the modality gap. What KD reliably changes is calibration, and per-sample, label-anchored gating is the change that turns its gains statistically significant.

## Results — 5-fold LOSO (ASPED v.a)

> **Status (July 2026).** A compute-server failure delayed the fold-level ECE and three PR-AUC aggregates. They will be posted here as soon as the server is restored, together with per-fold metric CSVs, and incorporated in the camera-ready version of the paper.


| Method | Macro Acc | No-Ped | Ped | F1_Ped | *d* | *p* |
|--------|:---------:|:------:|:---:|:------:|:---:|:---:|
| Baseline (CE only) | 72.9 ± 1.8 | 63.5 | 82.3 | 0.286 | – | – |
| LogitKD | 73.6 ± 1.7 | 70.8 | 76.4 | 0.312 | 1.18 | 0.079 |
| LogitKD (α=0.3) | 73.8 ± 1.9 | 70.3 | 77.2 | 0.309 | 1.36 | 0.053 |
| LogitKD + EMA loss-norm | 73.6 ± 1.9 | 68.5 | 78.7 | 0.304 | 1.01 | 0.086 |
| LogitKD + logit adjustment | 71.3 ± 2.1 | 53.7 | **89.0** | 0.259 | −0.94 | 0.104 |
| LogitKD + Focal | 73.3 ± 2.0 | 65.3 | 81.3 | 0.293 | 0.50 | 0.330 |
| Cosine | 73.6 ± 1.7 | 67.4 | 79.8 | 0.301 | 1.04 | 0.106 |
| CRD | 73.2 ± 2.0 | 64.4 | 82.0 | 0.290 | 0.30 | 0.579 |
| RKD | 73.7 ± 1.6 | 67.3 | 80.1 | 0.300 | 1.08 | 0.096 |
| **LogitKD$_\mathrm{TFD}$** | **73.7 ± 1.7** | 72.6 | 74.7 | **0.318** | 1.60 | **0.033*** |
| **Hybrid$_\mathrm{TFD}$** | 73.6 ± 1.8 | **72.9** | 74.3 | 0.318 | **1.72** | **0.026*** |

Accuracies in percent, mean ± std across folds where shown. * marks *p* < 0.05 on paired per-fold *t*-tests against the baseline. LOSO PR-AUC stays within 0.326–0.332 for the logged configurations (baseline 0.328), so the gains are calibration-driven rather than ranking-driven (see paper, Sec. 5).

## Method in one equation

TFD multiplies the per-sample KD term by a binary mask that is zero only when the sample is minority-class (pedestrian) **and** the teacher misclassifies it. Up to a constant, the resulting objective is cross-entropy against a conditionally smoothed target: label smoothing that switches off exactly where it would point away from the ground truth (paper, Sec. 3.3).

## Quick start

```bash
conda create -n tfd python=3.9 && conda activate tfd
pip install -r requirements.txt

# Baseline (LOSO)
bash runners/run_baseline_loso.sh

# LogitKD + TFD (LOSO)
bash runners/run_kd_loso.sh configs/kd_logit_unw_immune.yaml   # 'immune' = TFD mask

# Aggregate 5-fold results + paired t-tests
python scripts/aggregate_loso.py
python scripts/loso_paired_tests.py
```

Note: config files use the internal name `immune` for the TFD mask (and `selective` for a confidence-gated variant not reported in the paper).

## Repository structure

```
models/       student (VGGish + Transformer), teacher head, KD losses
data/         ASPED datamodule, inverse-frequency weighted sampler
configs/      the eleven compared configurations + ablations
scripts/      teacher embedding extraction, LOSO aggregation, statistics
runners/      shell entry points for LOSO chains
```

Data: ASPED v.a audio/video and labels are available at the [ASPED dataset page](https://huggingface.co/datasets/urbanaudiosensing/ASPED). Teacher embeddings are extracted with `scripts/extract_embeddings.py`.

## Citation

Citation entry will be added upon publication (DCASE 2026, under review).

## License

MIT (see LICENSE).
