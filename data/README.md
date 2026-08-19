# Local clinical data

Patient-level clinical data from the University Medical Center Mannheim (UMM)
and Mainz (MAINZ) cohorts are intentionally absent from GitHub. These data are
controlled access; their availability and the procedure for requesting access
are described in the manuscript and in the associated heiDATA record. This
document describes only the repository's file contract and contains no patient
data.

## Raw inputs

The preprocessing entry points use paths relative to `src/`. Before running
them, an authorized user must place these three Excel workbooks directly in
this directory, with the names shown exactly:

```text
data/
├── 20231129 Lap und Histo Daten von Ines Tuschner.xlsx  # raw UMM data
├── 202403 Lap und Histo Daten von Ines Tuschner.xlsx    # raw UMM supplementary laboratory data
└── 20240813-FibrosisDB(302_Patients).xlsx               # raw MAINZ data
```

`src/preprocess.py` reads the two UMM `.xlsx` workbooks together and uses the
MAINZ `.xlsx` workbook as the external (`prospective`) cohort. The optional
fine-tuning path reverses those roles for its train/validation/test split and
uses a held-out UMM subset. Other utilities that inspect raw data refer to the
same exact workbook names. The legacy conversion utility can create
`20240813-FibrosisDB_converted.xlsx`, but that file is a local derivative and is
not an input read by the current preprocessing pipeline.

## Generated data contract

Preprocessing writes ordinary CSV files without an index. For each task
`<task>` (`fibrosis`, `two_stage`, `cirrhosis`, or `three_stage`) and split
`<split>`, the expected layout is:

```text
data/
├── preprocessed_no_mice_<split>/
│   └── <split>_<task>.csv
├── preprocessed_mice_fib_<split>/
│   ├── <split>_<task>_0.csv
│   ├── ...
│   └── <split>_<task>_9.csv
├── preprocessed_no_mice_data.csv
└── preprocessed_mice_fib_data.csv
```

The normal split names are `train`, `val`, `test`, and `prospective`; the last
is the MAINZ external cohort. Fine-tuning additionally uses `train_ft`,
`val_ft`, and `test_ft`. The files under `preprocessed_no_mice_<split>/` are
locally generated, cleaned/preprocessed tables that retain missing values. The
ten numbered files under `preprocessed_mice_fib_<split>/` are locally generated
multiple-imputation artifacts; they also contain the calculated FIB-4 and APRI
fields used by downstream evaluation and reporting scripts. Those scripts most
commonly load imputation `_0`, while model training can consume all ten.

The two top-level `preprocessed_*_data.csv` files are locally generated merged
tables assembled from the `train`, `val`, `test`, and `prospective` task files.
Because they are rewritten once per task, they represent the most recently
processed task rather than a complete archive of every task.

The standalone execution block in `src/preprocess.py` also has a legacy path
that can generate NumPy arrays `xs_train.npy`, `xs_test.npy`, `ys_train.npy`,
and `ys_test.npy`, plus the pickled column list `df_cols.pickle`. These are local
artifacts, not raw inputs for the current training entry point.

## Handling controlled data

All contents of `data/` remain ignored by Git except this README. Authorized
users must copy the approved files into their own local checkout and keep raw,
preprocessed, imputed, converted, and legacy artifacts untracked. Do **not** use
`git add --force` (or otherwise override the ignore rules) to commit any of
these files.
