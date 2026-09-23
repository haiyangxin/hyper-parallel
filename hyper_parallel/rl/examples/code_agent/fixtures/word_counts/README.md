# Word counting

Repair word counting. Input is a JSON list of strings. Count case-insensitive ASCII words matching [a-z]+ across all strings; punctuation, digits and whitespace separate words. Return a JSON object mapping each lowercase word to its count. Empty input returns {}. Only src/*.py may change, add new modules if useful; entrypoint.py and public_test.py are protected.

Run a JSON request:

```bash
python entrypoint.py < input.json
```

Run public checks:

```bash
python public_test.py
```
