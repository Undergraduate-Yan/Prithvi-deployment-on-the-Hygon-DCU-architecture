"""Export the retained flood model using an explicitly located study input tree."""
import argparse
import os
from pathlib import Path
import runpy

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True,
                        help='Root containing the frozen flood data, checkpoint and evaluation metadata; see docs/data_and_models.md')
    args = parser.parse_args()
    os.environ['PRITHVI_FLOOD_ROOT'] = str(args.project_root.resolve(strict=True))
    runpy.run_path(str(Path(__file__).with_name('export_onnx_fp32.py')), run_name='__main__')

if __name__ == '__main__':
    main()
