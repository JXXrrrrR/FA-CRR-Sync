# FA-CRR-Sync

A PyTorch implementation of **Foreground-Aware Contrastive Regression with
Replay (FA-CRR-Sync)** for synchronized-diving action quality assessment.

FA-CRR-Sync uses Live and Replay1 RGB streams together with their foreground
masks. RGB and mask I3D features are fused by sigmoid gating, then processed by
the contrastive regression with replay (CRR) framework for score and
synchronization assessment.

<p align="center">
  <img src="images/pipeline.png" alt="FA-CRR-Sync pipeline" width="90%" />
</p>

<p align="center">
  <img src="images/regression.png" alt="FA-CRR-Sync regression framework" width="90%" />
</p>

## Dataset 📦

Place the companion `FineSynchro_Final_Data` directory next to this repository:

```text
parent_directory/
  FA-CRR-Sync/
  FineSynchro_Final_Data/
    annotations/
    rgb/live/
    rgb/replay1/
    masks/live/
    masks/replay1/
    metadata/
```

The default configuration is [`configs/data/finesynchro.yaml`](configs/data/finesynchro.yaml).
It uses the 450/50/165 train/validation/test split and samples 96 frames per
stream. RGB frames and masks receive identical spatial transforms.

## Installation ⚙️

Use Python 3.10 and install the package from the repository root:

```bash
python -m pip install -e .
```

The provided environment snapshot is available in
[`configs/environment/altolia_v1.json`](configs/environment/altolia_v1.json).

## Training 🚀

Train FA-CRR-Sync with the paper-aligned configuration:

```bash
python scripts/train_crr_sync.py \
  --config configs/experiments/fa_crr_sync.yaml
```

The configuration uses RGB I3D initialization from
`checkpoints/i3d_kinetics400_rgb.pth`, fixed same-action references, and
ten-reference inference.

## Evaluation 📊

Training selects the checkpoint on the validation set and evaluates the selected
checkpoint on the test set. Outputs are written to the run directory specified
by the experiment configuration.

For a lightweight local data check:

```bash
python scripts/smoke_data.py
python -m unittest discover -s tests -v
```

## Repository layout

```text
FA-CRR-Sync/
  checkpoints/       I3D initialization checkpoint
  configs/           dataset and experiment configurations
  data/              manifests, split files, and validation reports
  images/            README figures
  scripts/           training and utility scripts
  src/fa_crr_sync/   model, data, evaluation, and training modules
  tests/             unit tests
```
