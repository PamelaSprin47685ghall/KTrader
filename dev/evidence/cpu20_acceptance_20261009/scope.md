# CPU 2.0 acceptance; GPU 2.1

User scope change on 2026-10-09: GPU is not a 2.0 release blocker. This
iteration closes CPU evidence without altering runtime source, tests,
dependencies or numerical tolerances. No new GPU work is performed.

The existing earlier-window verifier now ran twice through its original
entry:24 assertions for run1;24 chronological-model assertions plus4
independent-process comparisons for run2. All passed. Every daily portfolio
L1 difference was0, as was the mutual return difference. Producer1's old
postprocessing timeout remains124; the recovered numeric output is not
relabeled as a successful producer command. Producer2 and both verifier
commands actually returned0.

`dev/cpu20_acceptance.jl` is a separate stdlib-only release evidence audit,
not a new solver or a replacement test suite. It rechecks the preceding
41-file/22-group snapshot and receipts, raw window bytes, input hashes,
current runtime identities, and the completed verifier receipts/logs. It
also compares the already-saved portfolios. It must reject missing evidence
or changed qualified source. Its own self-tests are evidence-checker tests,
not additional KTrader numerical tests or a single-process full-suite run.

Acceptance refers to the local CPU engineering delivery at the documented
single-task/batch configuration. It does not prove arbitrary-history
convergence, performance optimality, long-horizon throughput, or safe memory
headroom for two or more concurrent date tasks. No version metadata, commit,
tag, remote release, trading deployment or system thread default is changed.
