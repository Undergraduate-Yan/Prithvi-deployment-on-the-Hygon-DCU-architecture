"""Locate research modules without installing or changing the host environment."""
from pathlib import Path
import ast
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]

def show_source_help(target, program):
    """Describe the CLI without importing numerical or accelerator libraries.

    Show source declarations, including required flags, choices and defaults.
    Expressions remain literal: evaluating them could initialize a runtime.
    """
    source = target.read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    print(f'Usage: python {program} [arguments]')
    print(ast.get_docstring(tree) or target.name)
    print('\nArgument declarations from the program source (no runtime imports):')
    calls = sorted((node for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ('add_argument', 'add_subparsers', 'add_parser')),
                   key=lambda node: (node.lineno, node.col_offset))
    for node in calls:
        print('  ' + ' '.join(ast.get_source_segment(source, node).split()))
    if not calls:
        print('  No argparse declarations; consult the module documentation.')

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
    target = ROOT / routes[action]
    arguments = sys.argv[2:]
    option_arguments = arguments[:arguments.index('--')] if '--' in arguments else arguments
    if any(value in ('-h', '--help') for value in option_arguments):
        show_source_help(target, Path(sys.argv[0]).name + ' ' + action)
        return
    configure()
    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name='__main__')
