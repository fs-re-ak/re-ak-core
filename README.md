# re-ak-core
Open core shared by both the NucleusKit toolkit and online processor

## Repository content

The `reak_core` package itself arrives with the extraction phase. For now the repository holds the material used to
check it against the two current systems:

| Path | Content |
|---|---|
| `tests/synthetic/` | Deterministic synthetic session generator (Hermès EXG, Shimmer PPG / GSR, events) and the catalogue of cases |
| `tests/references/` | Reference outputs of the current NucleusKit and Prime pipelines on those cases, and the measured differences between them ([README](tests/references/README.md)) |
| `tools/` | Recording (`record_references.py`), comparison (`report_divergences.py`) and the shared toolkit used by the non-regression tests of both systems (`refkit.py`) |
| `requirements-reference.txt` | Pinned environment the references were recorded with |

No real participant data is stored in this repository.

## License

Copyright (C) 2026 RE-AK Technologies Inc.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version. See [LICENSE](LICENSE) for the full text.

SPDX-License-Identifier: `GPL-3.0-or-later`. Every source file carries this identifier in its header; vendored
third-party files keep their original header and license.
