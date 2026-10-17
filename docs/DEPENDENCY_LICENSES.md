# Dependency licenses

Snapshot of 2026-09-25. Not legal advice, and not a clearance: it lists what the installed packages declare about themselves. Regenerate the inventory below with `python scripts/license_inventory.py --output ...`; the findings and limits sections are written by hand.

## Inventory (generated)

Closure read from installed metadata: 221 distributions (77 declared in a manifest, 144 transitive). 214 permissive, 5 weak copyleft, 1 copyleft, 1 unresolved.

| Distribution | Version | Origin | Declared license | Category | Note |
|---|---|---|---|---|---|
| hopcroftkarp | 1.2.5 | transitive, via persim | GNU General Public License v3 (GPLv3) ; GPL | copyleft |  |
| asyncssh | 2.24.0 | declared | OSI Approved ; EPL-2.0 OR GPL-2.0-or-later | weak copyleft | dual: the holder chooses the loosest option |
| certifi | 2026.6.17 | transitive, via requests | Mozilla Public License 2.0 (MPL 2.0) ; MPL-2.0 | weak copyleft |  |
| orjson | 3.11.9 | transitive, via ccxt | MPL-2.0 AND (Apache-2.0 OR MIT) | weak copyleft |  |
| pathspec | 1.1.1 | transitive, via mypy | Mozilla Public License 2.0 (MPL 2.0) | weak copyleft |  |
| tqdm | 4.70.0 | transitive, via huggingface-hub | MPL-2.0 AND MIT | weak copyleft |  |
| streamlit-autorefresh | 1.0.1 | declared | (nothing declared) | unresolved | MIT: metadata says UNKNOWN, but the shipped LICENSE is the MIT text (with the packaging template's copyright line) and the repository declares MIT. |

Declared metadata that is misleading, reviewed by hand:

- ib-async 1.0.3: BSD 2-Clause: the shipped LICENSE and the repository say so; the 'Other/Proprietary' classifier is a labelling error.
- v20 3.0.25.0: MIT (copyright OANDA): the shipped LICENSE and the repository say so; the 'Other/Proprietary' classifier is a labelling error.

In the closure but not installed here (Linux-only, GPU or optional), so not read: coincurve (via ccxt), cuda-bindings (via torch), cuda-toolkit (via torch), hf-xet (via huggingface-hub), nvidia-cudnn-cu13 (via torch), nvidia-cusparselt-cu13 (via torch), nvidia-nccl-cu12 (via xgboost), nvidia-nccl-cu13 (via torch), nvidia-nvshmem-cu13 (via torch), pytest-cov (via a manifest), triton (via torch), uvloop (via uvicorn), watchfiles (via uvicorn).


## Findings (by hand)

- **`hopcroftkarp` is GPLv3** (the license file in the package says so). It arrives through `ripser` -> `persim` -> `hopcroftkarp`. `ripser` is declared in `requirements.txt`, and only `nothierly_genesis/` imports it (`tda_engine.py`, `topological_crash_detector.py`) or `persim` (`plot_persistence.py`). A GPLv3 obligation arises when software that combines it is distributed. The simple remedy: keep `nothierly_genesis/` out of the perimeter of a sale, and remove `ripser` from `requirements.txt` or replace it.
- **`asyncssh` is offered as EPL-2.0 or GPL-2.0-or-later.** The EPL-2.0 is the one to choose. It is used only in `src/mlops/ssh_orchestrator.py`, not in the execution core.
- **MPL-2.0** (`certifi`, `orjson`, `tqdm`, and `pathspec`, which comes with the `mypy` dev tool): obligations are file-level and apply to changes made to those packages' own files. None is modified here.
- **Wrong or missing metadata, checked by hand** against the shipped license file and the project page: `ib-async` (BSD 2-Clause), `v20` (MIT, copyright OANDA) and `streamlit-autorefresh` (MIT text with the packaging template's copyright line). `tiktoken` puts its whole MIT text in the `License` field, which the script now reads from its title line.
- Using the Interactive Brokers and OANDA services requires accepting their own terms of service. That is a contract matter, not a license of the Python packages.

## Limits

- Only declared metadata was read. Source files and the native libraries bundled in wheels were not scanned (for example `libzmq` in `pyzmq`, understood to be LGPL with a static-linking exception, not verified here).
- Thirteen distributions in the closure are not installed in the environment that produced this and could not be read: `coincurve`, `cuda-bindings`, `cuda-toolkit`, `hf-xet`, several `nvidia-*` CUDA wheels, `pytest-cov`, `triton`, `uvloop` and `watchfiles`. The NVIDIA CUDA packages carry NVIDIA's own license. `requirements.txt` points `torch` at the CPU-only index, so they should not be installed in the image; confirm by building it.
- This says nothing about the origin of the code itself: no third-party copyright or license header was found in the tracked source, which shows the absence of headers, not the absence of copied code.
- A scanner (ScanCode, FOSSA) and counsel are the next step before a sale.
