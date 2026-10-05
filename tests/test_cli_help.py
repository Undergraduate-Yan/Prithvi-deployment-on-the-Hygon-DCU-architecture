"""Check command help with site-packages disabled, without importing runtimes."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def routes():
    for path in sorted((ROOT / 'scripts').glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dispatch':
                yield path, ast.literal_eval(node.args[0])


class CliHelpTests(unittest.TestCase):
    def invoke(self, script, *args):
        return subprocess.run([sys.executable, '-S', str(script), *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=20)

    def test_all_dispatch_routes_without_site_packages(self):
        count = 0
        for script, mapping in routes():
            for action in mapping:
                for option in ('--help', '-h'):
                    with self.subTest(script=script.name, action=action, option=option):
                        result = self.invoke(script, action, option)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn('Usage:', result.stdout)
                        self.assertIn('add_argument(', result.stdout)
                count += 1
        self.assertGreaterEqual(count, 32)

    def test_reported_cloud_help_contains_real_required_options(self):
        result = self.invoke(ROOT / 'scripts/evaluate.py', 'cloud', '--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ('--model', '--cache', '--payload-root', '--input-contract', '--output-root', '--role'):
            self.assertIn(flag, result.stdout)
        self.assertIn('required=True', result.stdout)

    def test_unknown_action_fails_even_with_help(self):
        result = self.invoke(ROOT / 'scripts/evaluate.py', 'unknown-action', '--help')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unknown action', result.stderr)

    def test_top_level_help(self):
        result = self.invoke(ROOT / 'scripts/evaluate.py', '--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('cloud', result.stdout)


if __name__ == '__main__':
    unittest.main()
