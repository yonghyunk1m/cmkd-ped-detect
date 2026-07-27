# LOSO prediction dumps

Per-fold student predictions (`{session_<id>_probs, session_<id>_labels}`) for the four
configurations whose raw outputs back the paper's verified numbers:
`baseline_binary_v1`, `kd_logit_unw` (LogitKD), `kd_logit_unw_immune` (LogitKD_TFD),
`kd_hybrid_logit_cosine_immune` (Hybrid_TFD).

The `.npz` files (~35 MB each) are not tracked in git to keep the repo light. They are
available on request, and can be regenerated from the released checkpoints with
`inference.py` (see the ECE reproduction guide in the top-level README). Once placed in
this directory, `python scripts/verify_paper_numbers.py` reproduces Table 1, the
significance test, and the calibration numbers exactly.
