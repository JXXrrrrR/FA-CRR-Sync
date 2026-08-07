# Script entry points

- `build_release_manifest.py`: build and hash the 450/50/165 split.
- `audit_release_data.py`: scan all release sample paths, frame formats, counts,
  and RGB/mask alignment.
- `run_release_data_audit_background.ps1`: run the complete data audit in a
  hidden background process and save stdout, stderr, and the exit code.
- `smoke_data.py`: load one real Live/Replay pair and both masks through the
  exact 96-frame preprocessing contract.
- `train_crr_sync.py`: train, validate, select, and finally test either the
  mask-free CRR-Sync baseline or FA-CRR-Sync.
- `analyze_dataset.py`: summarize split, action, score, difficulty, and source
  statistics.
- `audit_masks.py`: inspect a deterministic mask sample.
- `analyze_predictions.py`: calculate bootstrap intervals and paired analyses
  from saved per-sample predictions.
- `smoke_train_step.py`: GPU optimization smoke test.
