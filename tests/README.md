# Test strategy

The first required tests cover:

1. canonical sample-ID and 450/50/165 split loading;
2. Live/Replay and dual-mask frame sampling;
3. score and synchronization-label parsing;
4. SRCC, MAE, R-l2, and AIoU metrics;
5. deterministic reference selection;
6. one-batch model forward/backward;
7. checkpoint and per-sample prediction round trips;
8. one-channel mask I3D and manuscript sigmoid-gating contracts.
