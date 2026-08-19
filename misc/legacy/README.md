# Legacy migration scripts

These scripts are retained as historical, one-off source migrations. They are
not part of the training or evaluation pipeline and should not normally be run:

* `apply_patch_comparators.py` added FIB-4 and APRI comparator support that is
  already present in `src/utils/validation_tools.py`.
* `apply_patch_fib4.py` is the earlier FIB-4-only migration and is superseded by
  both the comparator migration and the current implementation.

The original `src/` entry points remain as compatibility wrappers. Invoke those
wrappers from `src/`, as before, when reproducing the migration; they preserve
arguments and the caller's working directory.

* `shap_publication_figures_v1.py` is the pickle-only predecessor of the final
  manuscript SHAP generator. It has no tracked importer or required path role;
  use `src/shap_publication_figures.py` instead. The archived script retains its
  original algorithms and fixed output names for provenance only.
