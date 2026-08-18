# Standalone diagnostics

No audited diagnostic was moved here: the current QC scripts consume pipeline
modules, checkpoints, or established output paths, so keeping them in `src/`
avoids compatibility risk. This directory is reserved for future diagnostics
that are conclusively standalone.
