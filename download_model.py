"""Download the pinned CLIPSeg checkpoint into models/clipseg for offline use."""
import json
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
from text_protection import MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parent
FILES = ['config.json', 'preprocessor_config.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'vocab.json', 'merges.txt', 'model.safetensors']


def main():
    destination = ROOT / 'models' / 'clipseg'
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        print(f'[Model] {name}', flush=True)
        cached = hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION,
                                 cache_dir=ROOT / '.cache' / 'clipseg')
        shutil.copy2(cached, destination / name)
    (destination / 'source.json').write_text(json.dumps(
        {'repository': MODEL_ID, 'revision': MODEL_REVISION,
         'license': 'apache-2.0', 'weights_tracked_in_git': False,
         'local_storage': 'regular files after running download_model.py'}, indent=2))
    print(f'[Ready] {destination}', flush=True)


if __name__ == '__main__':
    main()
