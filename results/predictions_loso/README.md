# LOSO prediction dumps

Per-fold student predictions (`{session_<id>_probs, session_<id>_labels}`) for the four
configurations whose raw outputs back the paper's verified numbers:
`baseline_binary_v1`, `kd_logit_unw` (LogitKD), `kd_logit_unw_tfd` (LogitKD_TFD),
`kd_hybrid_logit_cosine_tfd` (Hybrid_TFD).

The four `.npz` files (~35 MB each) are committed via Git LFS, so they ship with the repo
(install `git lfs` to pull the actual arrays). `python experiments/reproduction/recompute_from_npz.py`
reproduces Table 1, the significance tests, and the calibration numbers directly from them.
They can also be regenerated from the released checkpoints with `inference.py` (see the ECE
reproduction guide in the top-level README).
