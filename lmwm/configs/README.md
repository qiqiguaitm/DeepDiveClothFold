# LMWM Configs

Keep one YAML per experiment family:

- `datasets/*.yaml`: dataset roots, CRAVE artifact paths, split rules.
- `models/*.yaml`: state world model architecture and loss choices.
- `training/*.yaml`: optimizer, batch size, schedule, logging, checkpoint paths.

Configs should be copied into each run log directory before training, matching
the `kai0` and LaWAM run-management style.
