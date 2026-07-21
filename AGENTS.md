# Project change-record policy

For every change to the manuscript, study design, analysis method, code, data
schema, generated result, figure, or documentation in this repository, append a
corresponding entry to `docs/研究审查与修订总账.md` before finishing the task.

Each entry must state:

- date and revision ID;
- affected files;
- what changed and why;
- evidence or validation performed;
- remaining limitations or required follow-up.

The revision table is append-only. Do not rewrite or delete prior entries. If a
prior entry is wrong, add a corrective entry that cites the earlier revision ID.

Never present an unexecuted analysis, placeholder, or planned experiment as an
observed result. Distinguish clearly among reproduced legacy results, newly
executed results, and pending analyses.

## Branch and research-standard policy

Treat `original` as an immutable audit baseline. Development on `main` is not
constrained by the architecture, package choices, interfaces, hyperparameters,
or implementation details of the legacy scripts. Reuse legacy code only when
independent evidence shows that doing so remains methodologically justified.

Design every `main`-branch method toward top-journal evidentiary standards:
pre-specified hypotheses and analysis units, leakage-safe validation, strong
and transparent baselines, ablations and falsification controls, uncertainty
at the correct independence level, reproducible environments and artifacts,
and conclusions bounded by the available data. When the current dataset cannot
support a top-journal claim, preserve the standard and narrow the claim; never
lower the validation standard or hide a data limitation to retain a result.

Project dependencies may be installed into the repository-local `.venv` when
needed to implement or validate an authorized algorithm change. Record new
runtime dependencies in the applicable lock file and in the append-only Chinese
revision ledger; do not rely on undocumented global packages.
