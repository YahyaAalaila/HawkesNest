# Output Schema

HawkesNest exports event data as CSV, JSONL, and metadata.

## CSV

CSV exports use one row per event:

```text
t,x,y,m,is_triggered
```

- `t`: event time.
- `x`, `y`: spatial coordinates.
- `m`: mark label.
- `is_triggered`: whether the accepted event had nonzero triggering intensity.

## JSONL

JSONL exports store one sequence per line. Each JSONL row includes:

- `times`: event times.
- `locations`: event spatial coordinates as `[x, y]` pairs.
- `marks`: mark labels.
- `is_triggered`: whether each accepted event had nonzero triggering intensity.

Example:

```json
{"times": [0.1, 0.3], "locations": [[0.2, 0.4], [0.8, 0.1]], "marks": [1, 2], "is_triggered": [false, true]}
```

## Metadata

Metadata exports include suite, level, seed, event counts, tau window, simulator class, configuration, and export paths.
