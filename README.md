# re-ak-core
Open core shared by both the NucleusKit toolkit and online processor

`reak_core` processes recordings of the Hermès headset (EEG and facial EMG) and of the Shimmer wristband (EDA and
PPG): engagement, emotions, arousal and heart dynamics on a shared 2 Hz timebase. It is the single implementation
used by the free Nucleus-Kit toolkit (local processing) and by the re-akcas cloud processor.

Status: **1.0.0a1, pre-release**. See [CHANGELOG.md](CHANGELOG.md) for every behaviour change compared with the
previous NucleusKit and Prime pipelines, measured on synthetic data.

## Install

```bash
pip install "re-ak-core @ git+https://github.com/fs-re-ak/re-ak-core.git"
# development
pip install -e ".[dev]"
```

Python ≥ 3.10. scikit-learn is pinned to the 1.6 series: the emotion models were saved with 1.6.1.

## Use

```bash
reak-core --session path/to/session            # default pipeline, model classical-emotion-2.6.0
reak-core --session path/to/session --model classical-emotion-2.5.0 --force
reak-core --list-models
```

```python
from reak_core.pipeline import Pipeline

result = Pipeline.default(emotion_model="standard-latest").run("path/to/session")
print([(step.name, step.status) for step in result.steps])
```

A session folder holds `rawData/` (Vizia Mobile or Nucleus-Kit files: `eeg.csv`, `shimmer.csv`, `event.csv`, …). The
pipeline writes `results/` (`Emotions.csv`, `Cognition.csv`, `Arousal.csv`, `HeartDynamics.csv`, `emotion_model.json`,
`processing.json`), diagnostic files and figures under `features/`, and reuses existing outputs unless `force=True`.
`results/processing.json` records the core version, the model, each step's status and a signal-quality summary.

The modules can also be used one by one: `reak_core.io` (loading, invalid samples), `reak_core.femg`
(`compute_emotions`), `reak_core.eeg` (`compute_cognition`), `reak_core.eda` (`compute_arousal`), `reak_core.ppg`
(`compute_heart_dynamics`), `reak_core.models` (`load_model`), `reak_core.tools` (channel fixer, PPG fixer).

## Extend

Other packages add behaviour without modifying the core:

| Entry point group | Provides | Example |
|---|---|---|
| `reak_core.steps` | a callable returning a `reak_core.pipeline.Step` | an extra analysis written to `features/` |
| `reak_core.exporters` | a callable returning a `reak_core.pipeline.Exporter` | a report or an upload after the steps |
| `reak_core.models` | a callable `factory(key) -> EmotionModel` | another emotion model, selectable by key |

`Pipeline.default().with_installed_extensions()` (or `reak-core --extensions`) adds installed steps and exporters;
`with_step(step, after="emotions")` adds one explicitly.

## Repository content

| Path | Content |
|---|---|
| `src/reak_core/` | The package; model releases in `models/releases/` (checksummed, see each `manifest.json`) |
| `tests/core/` | Unit tests and the comparison of the core with the references, output by output (`expectations.py`) |
| `tests/synthetic/` | Deterministic synthetic session generator (Hermès EXG, Shimmer PPG / GSR, events) and the catalogue of cases |
| `tests/references/` | Reference outputs of NucleusKit 1.1 and Prime 3.1.0 on those cases, and the measured differences between them ([README](tests/references/README.md)) |
| `tools/` | Recording (`record_references.py`), comparison (`report_divergences.py`), CHANGELOG measurement (`write_changelog.py`) and the shared toolkit used by the non-regression tests of both systems (`refkit.py`) |
| `requirements-reference.txt` | Pinned environment the references were recorded with |

No real participant data is stored in this repository.

## License

Copyright (C) 2026 RE-AK Technologies Inc.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version. See [LICENSE](LICENSE) for the full text.

SPDX-License-Identifier: `GPL-3.0-or-later`. Every source file carries this identifier in its header; vendored
third-party files keep their original header and license (`src/reak_core/eda/cvxEDA.py`: Luca Citi and Alberto
Greco, GPL-3.0-or-later).
