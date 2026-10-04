"""Use the study diagnostic aggregator to plot complete per-scene block records."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    target = root / 'src/diagnostics/aggregate_blocks.py'
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name='__main__')
