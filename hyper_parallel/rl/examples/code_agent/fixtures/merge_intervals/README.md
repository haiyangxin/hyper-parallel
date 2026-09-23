# Interval merging

Repair interval merging. Input is a JSON list of [start, end] integer intervals (start <= end). Return sorted disjoint intervals; touching intervals must merge. Handle empty input, unsorted input, nesting and duplicates. Only src/*.py may change; entrypoint.py and public_test.py are protected.

Run a JSON request:

```bash
python entrypoint.py < input.json
```

Run public checks:

```bash
python public_test.py
```
