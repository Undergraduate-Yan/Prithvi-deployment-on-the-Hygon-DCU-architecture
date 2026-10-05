# Metric contracts and command help

Run from the repository root with Python and NumPy installed:

```sh
python -m unittest discover -s tests -v
```

The tests cover hand-computed flood metrics, ignored and empty pools, rejected labels and shapes, direct confusion-count calls, consistency between the two flood metric modules, retained integer-count aggregates, and every dispatched CLI action with Python site-packages disabled. The shared dispatcher reads argparse declarations from source for `ACTION --help`; required options, choices and default expressions are displayed without importing model or accelerator libraries.

These checks do not validate K100 execution, model artifacts, or complete end-to-end reproduction. See `docs/external_requirements.md` for those prerequisites.
