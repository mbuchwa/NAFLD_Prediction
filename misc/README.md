# Maintainer references and archived utilities

* [Python module execution and data-flow audit](python_module_audit.md) records the role, imports, invocation contract, data/checkpoint requirements, outputs, execution mode, downstream consumers, and relocation decision for each audited Python entry point.
* [`legacy/`](legacy/) contains superseded, one-off source migration implementations. Compatibility wrappers remain at their original `src/` entry points.
* [`exploratory/`](exploratory/) contains standalone ad-hoc analyses. Compatibility wrappers are retained where historical invocation may still matter.
* [`diagnostics/`](diagnostics/) is reserved for genuinely standalone diagnostic utilities; the audit did not identify a current candidate whose move was worth the compatibility cost.

Files that are still imported, participate in dynamic dispatch, consume established pipeline paths, or may be externally imported remain in `src/` and are documented in the audit instead of being moved cosmetically.

* [`../SHAP_PROVENANCE.md`](../SHAP_PROVENANCE.md) identifies the canonical manuscript SHAP entry point and compares the retained historical workflows.
