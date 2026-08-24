# Primary-pipeline dependency and environment audit

This note compares the imports reachable from every verified stage in
`scripts/run_primary_pipeline.sh` with the two checked-in environment files. It
is an inventory, not a request to resolve or modernize dependencies. In
particular, no version in either specification has been changed.

## Canonical environment and exact creation commands

`environment.yml` is the canonical frozen environment. Its declared Conda
environment name is **`nafl`**. From the repository root, create and activate
that environment exactly as follows:

```bash
conda env create -f environment.yml
conda activate nafl
```

The YAML is an exported Linux/Python 3.11 environment with exact Conda package
builds and a small `pip:` subsection. Preserve those pins and builds when
reproducing the recorded environment; do not substitute versions from current
package indexes. The exported `prefix` is provenance from the maintainer's
machine, not a portable path that users should create directly.

## Audit scope and method

The primary workflow is the set of `train`, `evaluate`, `three-stage`, `shap`,
`reduced`, `clinical-utility`, `tables`, and `figures` entry points dispatched
by `scripts/run_primary_pipeline.sh`. Static AST inspection followed their
repository-local imports transitively. Standard-library and repository-local
modules were excluded. Distribution names were normalized where an import name
differs (`sklearn` to `scikit-learn`, for example).

Most imported distributions are present in `environment.yml`, including
NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn, openpyxl, XGBoost,
PyTorch, Pyro, and tqdm. The following imported dependencies are **absent from
the canonical environment specification**:

| Imported module | Distribution name | Primary-path evidence |
|---|---|---|
| `lightgbm` | `lightgbm` | LightGBM model implementation |
| `pytorch_lightning` | `pytorch-lightning` | FFN, TabTransformer, and shared neural-network implementations |
| `pytorch_tabular` | `pytorch-tabular` | GANDALF implementation and checkpoint validation/loading |
| `shap` | `shap` | canonical SHAP/validation path and MCMC-BNN implementation |
| `tab_transformer_pytorch` | `tab-transformer-pytorch` | shared TabTransformer network implementation |
| `te2rules` | `te2rules` | XGBoost implementation |
| `torchmetrics` | `torchmetrics` | VI-BNN and shared neural-network implementations |

“Absent” means that neither the Conda dependency list nor the YAML's `pip:`
subsection names the distribution. It does not assert that the package was
absent from the maintainer's live environment: a hand-installed package can be
importable without appearing in an incomplete export. Conversely, packages
that might arrive transitively are still listed above because the frozen file
does not declare them and therefore does not freeze their versions.

## Provenance of the additional packages

Repository history provides two distinct records:

* Commit `3686716` (2025-03-26, `initial commit`) introduced the unchanged
  `environment.yml` and the original, much shorter `requirements.txt`
  together. Neither file named the seven missing distributions. There is no
  checked-in post-creation install command, constraints file, or maintainer note
  from that snapshot establishing that they were intentionally added to the
  `nafl` environment through Conda or pip.
* Commit `d02190b` (2026-08-06, `update`) expanded `requirements.txt`. That later
  pip-oriented record explicitly names `lightgbm`, `pytorch-tabular`, `shap`,
  `tab-transformer-pytorch`, and `te2rules`. It does not explicitly name
  `pytorch-lightning` or `torchmetrics`; a pip resolver may install those as
  transitive requirements of the named neural packages, but this file neither
  pins nor independently records them.

Accordingly, the later requirements update is evidence that maintainers
intended a **separate pip installation snapshot** to cover five of the missing
direct imports. It is not evidence that those packages were deliberately
installed as an add-on to the original frozen `nafl` environment, and the
repository contains no comparable explicit record for the other two.

## Status of `requirements.txt`

`requirements.txt` is an **alternative/historical pip specification**, not a
numerically equivalent rendering of `environment.yml`. It records a different
PyTorch/CUDA combination and different versions of several overlapping
packages; it also omits many Conda runtime/build dependencies and leaves some
imported packages implicit. Do not install it on top of `nafl` and describe the
result as the canonical frozen environment. Use it only when deliberately
reconstructing that later pip snapshot, and report that choice separately.

