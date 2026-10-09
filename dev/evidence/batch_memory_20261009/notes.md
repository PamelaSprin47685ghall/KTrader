# Batch memory, 2026-10-09

Final measurements, source-qualified coverage and explicit unrun tests are
summarized in summary.md. qualification/status_partial.log is intentionally
not a full pass: no numeric receipt exists for the tool-blocked new file.

User asked for another substantial 2.0 iteration. Existing checkout/HEAD
602b897 retained. Initial read-only ps showed no Julia processes.
Initial source SHA256: backtest
087b1d0885d10918de7265bb14c6ffb23b5fdf7b4674f6d3ed8d71d95aae984d;
response 9fa232b1806026e271058af348af10f33024afd34df4b979732c3c698ee7f616.

Candidate: do not build an optional T×N×9 PrefixRulerStats table inside the
backtest when every consuming batch fit already verifies its entire prefix
with a full scan. Use the existing no-ruler-cache reference entry instead;
no guard is deleted and no public cache input is ignored. For incremental
backtests the internally built history cache is likewise never consumed
after initialize; no cache need be built for that backend. Before accepting
the candidate, compare rulers, full/fold statistics, posterior and portfolios
under the original numerical gates. Removing the cache changes accumulation
route (SIMD direct ruler vs sequential prefix sums), so exact values must
not be presumed. No solver, tolerance, fold, asset or history changes.

All command scopes retain 45s inner / 55s outer and sampled 2048MiB RSS
guard. No results prefilled; no commit/reset/clean/version change.

The first preparation microbenchmark hit the RSS guard (RC124,2116MiB).
Its numerical comparisons had printed17 passes, but the command did not
finish and is NOT a passing benchmark. Revised primitive measurement drops
each prepared result and explicitly collects between samples OUTSIDE their
timing; the accepted run completes17/17, RC0,17s,1919MiB. It is a controlled
primitive latency comparison, not natural-GC backtest throughput. Both
preparations allocate163006208 bytes; eliminating the prefix table saves
the separate100456200-byte resident payload, not those per-fit arrays.

The combined architecture + NEW backtest_cache_tests numeric command was
tool-blocked before execution with no approval token. It is not retried
through a different wrapper. That new test file remains UNRUN; existing
separate regression files and complete-window measurements are recorded
independently, not claimed as execution of the blocked tests.

Offline precompile printed successful KTrader image generation but the
parent command then timed out (RC124,35s,1662MiB). No successful-command
claim from it. Later existing-cache loads must succeed independently.
