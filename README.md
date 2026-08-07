# FA-CRR-Sync

Implementation of Foreground-Aware Contrastive Regression with Replay for
Synchronized-Diving Action Quality Assessment.

## Model

- Live and Replay1 RGB inputs, sampled to 96 frames;
- one-channel masks for Live and Replay1;
- RGB I3D and mask I3D backbones;
- sigmoid feature gating;
- three-stage procedure modeling;
- bidirectional relative score and synchronization regression;
- fixed same-action references and ten-reference inference.

## Layout

```text
FA_CRR_Sync_Final_Code/
  checkpoints/
  configs/
  data/
  scripts/
  src/
  tests/

FineSynchro_Final_Data/
  annotations/
  rgb/live/
  rgb/replay1/
  masks/live/
  masks/replay1/
  metadata/
```

## Commands

```powershell
C:\Users\11231\miniconda3\envs\Altolia_v1\python.exe -m unittest discover -s tests -v
C:\Users\11231\miniconda3\envs\Altolia_v1\python.exe scripts\smoke_data.py
C:\Users\11231\miniconda3\envs\Altolia_v1\python.exe scripts\train_crr_sync.py --config configs/experiments/fa_crr_sync.yaml
```
