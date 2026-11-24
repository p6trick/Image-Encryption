"""Submission entry point. Run without arguments for the coffee/cup CPU demo."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def execute(script, arguments):
    subprocess.run([sys.executable, '-s', str(ROOT / script), *arguments],
                   cwd=ROOT, check=True,
                   env={**os.environ, 'PYTHONNOUSERSITE': '1'})


def main():
    arguments = sys.argv[1:]
    command = arguments.pop(0) if arguments else 'demo'
    if command in ['-h', '--help', 'help']:
        print('python main.py [demo|examples|run|compare|restore|benchmark|compare-benchmarks|verify]')
        print('demo: coffee/cup protection and fresh-process restoration (default)')
        print('benchmark: RefCOCO 20 images, one-word input by default; --prompt-mode original also supported')
        print('run/compare/restore/examples: see python text_protection.py --help')
        return
    if command == 'demo':
        execute('text_protection.py', ['run', '--image', 'inputs/coffee.png', '--text', 'cup',
                                     '--output-dir', 'results/demo', *arguments])
    elif command == 'benchmark':
        if '--prompt-mode' not in arguments:
            arguments = ['--prompt-mode', 'word', *arguments]
        execute('evaluate_refcoco.py', arguments)
    elif command == 'compare-benchmarks':
        execute('compare_refcoco_prompts.py', arguments)
    elif command == 'verify':
        key = ROOT / 'results/verification/key.npz'
        execute('image_protection.py', ['train', '--key', str(key)])
        execute('verify_pipeline.py', [str(key), str(key.with_name('verification.json'))])
    elif command in ['run', 'compare', 'restore', 'examples']:
        execute('text_protection.py', [command, *arguments])
    else:
        raise SystemExit(f'Unknown command: {command}. Use python main.py --help')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
