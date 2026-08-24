# Post-restructure validation

Validation was performed from the repository root on 2026-08-24 without
running preprocessing, loading controlled data, loading checkpoints, or fitting
models. No dependencies were installed for these checks.

## Repository checks

| Check | Command or method | Result |
|---|---|---|
| Tracked Python syntax | `git ls-files -z '*.py' \| xargs -0 python -m py_compile` | Pass: every tracked Python file compiled. |
| Bash wrapper syntax | `git ls-files -z '*.sh' \| xargs -0 -r -n1 bash -n` | Pass: every tracked Bash file parsed. |
| Import-safe modules | A fresh Python process imported `scripts.check_environment`, `src.apply_patch_comparators`, `src.apply_patch_fib4`, `src.utils.ger_eng_dict`, and `src.variance_test_datasets`. | Pass: all five imports completed without data access or training. The compatibility wrappers only delegate under their `__main__` guards. |
| README module paths | Extracted `python -m src.<module>` commands and the `src/*.py` commands in the auxiliary table, converted them to repository paths, and checked them with `Path.is_file()`. | Pass: every documented code module exists. Generated output paths, path templates such as `<model>`, and controlled workbook prerequisites were classified as contracts rather than source modules. |
| Documented behavior | Parsed each README module with `ast` and checked for a `__main__` guard or a documented callable. | Pass: command modules have entry-point guards; `src.preprocess` also exposes the documented `prepare_data(...)` callable. |
| Repository-local imports | Parsed all tracked Python files with `ast`; resolved `src.*` imports from the repository root and the documented legacy `preprocess`/`utils` imports from `src/`. | Pass: no unresolved repository-local import target was found. Third-party availability is reported separately below. |
| Moved-file references | Searched tracked text for `apply_patch_comparators`, `apply_patch_fib4`, and `variance_test_datasets`, including their old and new paths. | Pass: executable references are the three compatibility wrappers, each of which targets an existing archived implementation. Remaining references are archive documentation or historical instructions inside the archived migration scripts. |
| Patch whitespace | `git diff --check` | Pass. |

## Environment limitations (not repository errors)

`python scripts/check_environment.py` reports that this validation interpreter
is Python 3.12.13 rather than the canonical Python 3.11 environment. It also
reports all 18 primary scientific dependencies, the optional `imblearn`
dependency, and the auxiliary `requests` and `yaml` imports as unavailable.
The controlled workbooks, generated cohort tables, and most model checkpoints
are intentionally absent. These conditions prevented broad imports of
scientific modules, but they do not indicate broken repository-local imports;
per instruction, no packages were installed and no pipeline stage was run to
work around them.

## Conclusion

The non-executing restructure checks found no repository error. Syntax,
wrapper parsing, safe imports, documented source paths and behaviors, local
import targets, moved-file compatibility references, and diff whitespace all
validated successfully. Scientific execution remains dependent on the pinned
environment, controlled data, and checkpoint contracts documented in the main
README.
