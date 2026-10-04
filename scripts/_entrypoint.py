"""Locate research modules without installing or changing the host environment."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]

def configure():
    paths = [ROOT / 'src'] + sorted(p for p in (ROOT / 'src').iterdir() if p.is_dir())
    for path in reversed(paths):
        sys.path.insert(0, str(path))

def dispatch(routes):
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python ' + Path(sys.argv[0]).name + ' ACTION [arguments]')
        for action, target in routes.items():
            print(f'  {action:22s} {target}')
        print('Use ACTION --help for the research program arguments.')
        return
    action = sys.argv[1]
    if action not in routes:
        raise SystemExit('Unknown action: ' + action)
    configure()
    target = ROOT / routes[action]
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name='__main__')
