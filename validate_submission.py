"""Run the submission workflow with networking disabled and save verification logs."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    validation = ROOT / 'validation'
    validation.mkdir(exist_ok=True)
    guard = validation / 'network_guard'
    guard.mkdir(exist_ok=True)
    (guard / 'sitecustomize.py').write_text(
        'import socket\n'
        'def denied(*args, **kwargs):\n'
        '    raise RuntimeError("Network disabled by submission verification")\n'
        'socket.socket.connect = denied\n'
        'socket.create_connection = denied\n')
    env = {**os.environ, 'PYTHONNOUSERSITE': '1', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'PYTHONPATH': str(guard),
           'PYTHONDONTWRITEBYTECODE': '1'}
    jobs = [
        ('verify', ['verify']),
        ('demo', ['demo']),
        ('benchmark_original', ['benchmark', '--prompt-mode', 'original']),
        ('benchmark_word', ['benchmark']),
        ('prompt_comparison', ['compare-benchmarks']),
        ('text_comparison', ['compare', '--image', 'inputs/coffee.png', '--texts', 'cup', 'spoon',
                             '--output-dir', 'results/cup_vs_spoon']),
    ]
    statuses = []
    for name, arguments in jobs:
        print(f'[START] {name}', flush=True)
        started = time.perf_counter()
        with (validation / f'{name}.log').open('w') as log:
            process = subprocess.run([sys.executable, '-s', str(ROOT / 'main.py'), *arguments],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        statuses.append({'name': name, 'returncode': process.returncode,
                         'seconds': time.perf_counter()-started})
        (validation / 'status.json').write_text(json.dumps(statuses, indent=2))
        if process.returncode:
            raise RuntimeError(f'{name} failed; see validation/{name}.log')
        print(f'[PASS] {name}', flush=True)
    original = json.loads((ROOT / 'results/refcoco20/summary.json').read_text())
    word = json.loads((ROOT / 'results/refcoco20_words/summary.json').read_text())
    assert abs(original['mean_iou'] - 0.20064452117625042) < 1e-6
    assert abs(word['mean_iou'] - 0.284914370394879) < 1e-6
    assert original['empty_predictions'] == 4 and original['exact_restorations'] == 16
    assert word['empty_predictions'] == 0 and word['exact_restorations'] == 20
    comparison = json.loads((ROOT / 'results/cup_vs_spoon/summary.json').read_text())
    assert comparison['comparisons'][0]['changed_mask_pixels'] > 0
    import numpy as np
    from PIL import Image
    from evaluate_refcoco import metrics
    example = metrics(np.array([[1,0],[1,0]],bool), np.array([[1,1],[0,0]],bool))
    assert example['iou'] == 1/3 and example['dice'] == .5
    for folder in ['refcoco20', 'refcoco20_words']:
        rows = json.loads((ROOT / 'results' / folder / 'per_sample.json').read_text())
        assert len(rows) == len({row['image_id'] for row in rows}) == 20
        for row in rows:
            directory = ROOT / row['directory']
            if row['restoration_verified']:
                assert row['max_pixel_error'] == 0 and row['mse'] == 0
                assert np.array_equal(np.array(Image.open(directory/'original.png')),
                                      np.array(Image.open(directory/'restored.png')))
            for name in ['ground_truth.png', 'mask.png', 'comparison.png']:
                Image.open(directory/name).verify()
    packages = ['torch','torchvision','transformers','numpy','Pillow','matplotlib',
                'pyarrow','pycocotools','requests','safetensors','scikit-image']
    report = {'all_passed': True, 'network_socket_connections_blocked': True,
              'python': sys.version, 'platform': sys.platform,
              'packages': {p: importlib.metadata.version(p) for p in packages},
              'jobs': statuses, 'original_summary': original, 'word_summary': word,
              'pixel_restoration_verified': True, 'twenty_unique_images_per_mode': True,
              'text_changes_mask': True,
              'scope': 'Linux CPU, existing school environment with user site disabled; not a clean-machine installation test'}
    (validation / 'verification.json').write_text(json.dumps(report, indent=2))
    print('[PASS] All submission workflows and assertions passed with network blocked', flush=True)


if __name__ == '__main__':
    main()
